"""
Code for Gaussian processes using GPflow and GPflowSampling.
"""

from argparse import Namespace
import copy
from re import A
import numpy as np
import tensorflow as tf
from gpflow import kernels
from gpflow.config import default_float as floatx
from gpflow.utilities import print_summary
import gpflow 
import random 

from .simple_gp import SimpleGp
from .gpfs.models import PathwiseGPR
from .gp.gp_utils import kern_exp_quad, kern_matern32, gp_post
from ..util.base import Base
from ..util.misc_util import dict_to_namespace, suppress_stdout_stderr
from ..util.domain_util import unif_random_sample_domain

import pandas as pd


class GpfsGp(SimpleGp):
    """
    GP model using GPFlowSampling.
    """

    def set_params(self, params):
        """Set self.params, the parameters for this model."""
        super().set_params(params)
        params = dict_to_namespace(params)

        # Set self.params
        self.params.name = getattr(params, 'name', 'GpfsGp')
        self.params.n_bases = getattr(params, 'n_bases', 1000)
        self.params.n_dimx = getattr(params, 'n_dimx', 1)
        self.set_kernel(params)

    def set_kernel(self, params):
        """Set GPflow kernel."""
        self.params.kernel_str = getattr(params, 'kernel_str', 'rbf')

        ls = self.params.ls
        kernvar = self.params.alpha**2

        if self.params.kernel_str == 'rbf':
            gpf_kernel = kernels.SquaredExponential(variance=kernvar, lengthscales=ls)
            kernel = getattr(params, 'kernel', kern_exp_quad)
        elif self.params.kernel_str == 'matern32':
            gpf_kernel = kernels.Matern32(variance=kernvar, lengthscales=ls)
            kernel = getattr(params, 'kernel', kern_matern32)
            # raise Exception('Matern 32 kernel is not yet supported.')

        self.params.gpf_kernel = gpf_kernel
        self.params.kernel = kernel

    def set_data(self, data):
        """Set self.data."""
        super().set_data(data)
        self.tf_data = Namespace()
        df = pd.DataFrame(self.data.x)
        # FIXME: Find better solution here
        if 'OpenML_task_id' in df.columns.values:
            df = df.drop('OpenML_task_id', axis = 1) # instance is not changed throughout a run; therefore, this is not a feature that is used by the model
        self.tf_data.x = tf.convert_to_tensor(df)
        self.tf_data.y = tf.convert_to_tensor(
            np.array(self.data.y, "float64").reshape(-1, 1)
        )
        self.set_model()

    def set_model(self):
        """Set GPFlowSampling as self.model."""

        # INFO: Original implementation does not do any hyperparameter optimization at all! 
        model_default = PathwiseGPR(
            data=(self.tf_data.x, self.tf_data.y),
            kernel=self.params.gpf_kernel,
            noise_variance=self.params.sigma**2,
        )

        try: 
            model = model_default
            opt = gpflow.optimizers.Scipy()
            res = opt.minimize(model.training_loss, model.trainable_variables)
            self.params.model = model

            print_summary(model)
        except:
            # go with default (no hyperparameter optimization)
            self.params.model = model_default
            print("GP likelihood optimization failed; go with default")


        # Check if it is a meaningful GP 
        # Predict on test data 
        d = self.data.x.copy()
        d = pd.DataFrame(d)
        if 'OpenML_task_id' in d.columns.values:
            d = d.drop('OpenML_task_id', axis=1)
        d = d.values.tolist()

        alpha = np.sqrt(model.kernel.variance.numpy())
        ls = list(model.kernel.lengthscales.numpy())
        sigma = np.sqrt(model.likelihood.variance.numpy())

        try: 
            mu, cov = gp_post(
                x_train=d,
                y_train=self.data.y,
                x_pred=d,
                ls=ls,
                alpha=alpha,
                sigma=sigma,
                kernel=self.params.kernel,
                full_cov=False
            )       
            lsratio = max(ls) / min(ls)

            if np.var(mu) < 10e-28 or lsratio > 10e3:
                print("Constant GP or ratio of lengthscales to high. Switch to default. ")
                self.params.alpha = 10.0
                self.params.ls = [2.5] * len(self.params.ls)
                self.params.sigma = 0.01
            else:
                self.params.alpha = alpha
                self.params.ls = ls
                self.params.sigma = sigma
        except: 
            self.params.alpha = 10.0
            self.params.ls = [2.5] * len(self.params.ls)
            self.params.sigma = 0.01
         

    def initialize_function_sample_list(self, n_samp=1):
        """Initialize a list of n_samp function samples."""
        n_bases = self.params.n_bases
        paths = self.params.model.generate_paths(num_samples=n_samp, num_bases=n_bases)
        _ = self.params.model.set_paths(paths)

        Xinit = tf.random.uniform(
            [n_samp, self.params.n_dimx], minval=0.0, maxval=0.1, dtype=floatx()
        )
        Xvars = tf.Variable(Xinit)
        self.fsl_xvars = Xvars

    @tf.function
    def call_fsl_on_xvars(self, model, xvars, sample_axis=0):
        """Call fsl on fsl_xvars."""
        fvals =  model.predict_f_samples(Xnew=xvars, sample_axis=sample_axis)
        return fvals

    def call_function_sample_list(self, x_list):
        """Call a set of posterior function samples on respective x in x_list."""

        # Model works not on dictionary but on list of lists
        x_list = [list(x.values()) if type(x) is dict else x for x in x_list]
        # Replace Nones in x_list with first non-None value
        x_list = self.replace_x_list_none(x_list)

        # Set fsl_xvars as x_list, call fsl, return y_list
        self.fsl_xvars.assign(x_list)
        y_tf = self.call_fsl_on_xvars(self.params.model, self.fsl_xvars)
        y_list = list(y_tf.numpy().reshape(-1))
        return y_list

    def replace_x_list_none(self, x_list):
        """Replace any Nones in x_list with first non-None value and return x_list."""

        # Set new_val as first non-None element of x_list
        new_val = next(x for x in x_list if x is not None)

        # Replace all Nones in x_list with new_val
        x_list_new = [new_val if x is None else x for x in x_list]

        return x_list_new

    # def gp_post_wrapper(self, x_list, data, full_cov=True):
    #     """Fits the Gaussian process and returns the posterior mean and standard deviation. All based on Gpflow."""

    #     """Wrapper for gp_post given a list of x and data Namespace."""
    #     if len(data.x) == 0:
    #         return self.get_prior_mu_cov(x_list, full_cov)


    #     d = data.x.copy()
    #     d = pd.DataFrame(d)
    #     if 'OpenML_task_id' in d.columns.values:
    #         d = d.drop('OpenML_task_id', axis=1)
    #     d = d.values.tolist()

    #     x = tf.convert_to_tensor(np.asarray(d, "float64"))
    #     y = tf.convert_to_tensor(
    #         np.array(data.y, "float64").reshape(-1, 1)
    #     )
    #     model = PathwiseGPR(
    #         data=(x, y),
    #         kernel=self.params.gpf_kernel,
    #         noise_variance=self.params.sigma**2,
    #     )

    #     xlarr = tf.convert_to_tensor(np.array(x_list, "float32")  / 1.0)


    #     model.kernel.variance = self.params.alpha**2
    #     model.kernel.lengthscales = self.params.ls

    #     # try: 
    #     #     opt = gpflow.optimizers.Scipy()
    #     #     res = opt.minimize(model.training_loss, model.trainable_variables)
    #     #     print_summary(model)
    #     # except:
    #     #     # go with default (no hyperparameter optimization)
    #     #     print("GP likelihood optimization failed; go with default")

    #     # Compute posterior
    #     try: 
    #         mu, cov = model.predict_f(xlarr)
    #     except: 
    #         print("DID NOT WORK")
    #         return self.get_prior_mu_cov(x_list, full_cov)

    #     mu = [el[0] for el in mu.numpy()]
    #     cov = [el[0] for el in cov.numpy()]


    #     return mu, cov



class MultiGpfsGp(Base):
    """
    Simple multi-output GP model using GPFlowSampling. To do this, this class duplicates
    the model in GpfsGp multiple times (and uses same kernel and other parameters in
    each duplication).
    """

    def __init__(self, params=None, data=None, verbose=True):
        super().__init__(params, verbose)
        self.set_data(data)
        self.set_gpfsgp_list()

    def set_params(self, params):
        """Set self.params, the parameters for this model."""
        super().set_params(params)
        params = dict_to_namespace(params)

        self.params.name = getattr(params, 'name', 'MultiGpfsGp')
        self.params.n_dimy = getattr(params, 'n_dimy', 1)
        self.params.gp_params = getattr(params, 'gp_params', {})

    def set_data(self, data):
        """Set self.data."""
        if data is None:
            self.data = Namespace(x=[], y=[])
        else:
            data = dict_to_namespace(data)
            self.data = copy.deepcopy(data)

    def set_gpfsgp_list(self):
        """Set self.gpfsgp_list by instantiating a list of GpfsGp objects."""
        data_list = self.get_data_list(self.data)

        # NOTE: GpfsGp verbose set to False (though MultiGpfsGp may be verbose)
        self.gpfsgp_list = [
            GpfsGp(self.params.gp_params, dat, False) for dat in data_list
        ]

    def initialize_function_sample_list(self, n_samp=1):
        """
        Initialize a list of n_samp function samples, for each GP in self.gpfsgp_list.
        """
        for gpfsgp in self.gpfsgp_list:
            gpfsgp.initialize_function_sample_list(n_samp)

    def call_function_sample_list(self, x_list):
        """
        Call a set of posterior function samples on respective x in x_list, for each GP
        in self.gpfsgp_list.
        """
        y_list_list = [
            gpfsgp.call_function_sample_list(x_list) for gpfsgp in self.gpfsgp_list
        ]

        # y_list is a list, where each element is a list representing a multidim y
        y_list = [list(x) for x in zip(*y_list_list)]
        return y_list

    def get_post_mu_cov(self, x_list, full_cov=False):
        """See SimpleGp. Returns a list of mu, and a list of cov/std."""
        mu_list, cov_list = self.gp_post_wrapper(x_list, self.data, full_cov)
        return mu_list, cov_list

    def gp_post_wrapper(self, x_list, data, full_cov=True):
        """See SimpleGp. Returns a list of mu, and a list of cov/std."""

        data_list = self.get_data_list(data)
        mu_list = []
        cov_list = []

        for gpfsgp, data_single in zip(self.gpfsgp_list, data_list):
            # Call usual 1d gpfsgp gp_post_wrapper
            mu, cov = gpfsgp.gp_post_wrapper(x_list, data_single, full_cov)
            mu_list.append(mu)
            cov_list.append(cov)

        return mu_list, cov_list

    def get_data_list(self, data):
        """
        Return list of Namespaces, where each is a version of data containing only one
        of the dimensions of data.y (and the full data.x).
        """
        data_list = []
        for j in range(self.params.n_dimy):
            data_list.append(Namespace(x=data.x, y=[yi[j] for yi in data.y]))

        return data_list
