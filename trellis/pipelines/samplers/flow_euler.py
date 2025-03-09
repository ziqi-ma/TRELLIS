from typing import *
import torch
import numpy as np
from tqdm import tqdm
from easydict import EasyDict as edict
from .base import Sampler
from .classifier_free_guidance_mixin import ClassifierFreeGuidanceSamplerMixin
from .guidance_interval_mixin import GuidanceIntervalSamplerMixin
import math
from trellis.models.structured_latent_flow import SLatFlowModel

class FlowEulerSampler(Sampler):
    """
    Generate samples from a flow-matching model using Euler sampling.

    Args:
        sigma_min: The minimum scale of noise in flow.
    """
    def __init__(
        self,
        sigma_min: float,
    ):
        self.sigma_min = sigma_min

    def _eps_to_xstart(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        return (x_t - (self.sigma_min + (1 - self.sigma_min) * t) * eps) / (1 - t)

    def _xstart_to_eps(self, x_t, t, x_0):
        assert x_t.shape == x_0.shape
        return (x_t - (1 - t) * x_0) / (self.sigma_min + (1 - self.sigma_min) * t)

    def _v_to_xstart_eps(self, x_t, t, v):
        assert x_t.shape == v.shape
        eps = (1 - t) * v + x_t
        x_0 = (1 - self.sigma_min) * x_t - (self.sigma_min + (1 - self.sigma_min) * t) * v
        return x_0, eps

    def _inference_model(self, model, x_t, t, cond=None, **kwargs):
        t = torch.tensor([1000 * t] * x_t.shape[0], device=x_t.device, dtype=torch.float32)
        return model(x_t, t, cond, **kwargs)

    def _get_model_prediction(self, model, x_t, t, cond=None, **kwargs):
        pred_v = self._inference_model(model, x_t, t, cond, **kwargs)
        pred_x_0, pred_eps = self._v_to_xstart_eps(x_t=x_t, t=t, v=pred_v)
        return pred_x_0, pred_eps, pred_v

    @torch.no_grad()
    def sample_once(
        self,
        model,
        x_t,
        t: float,
        t_prev: float,
        cond: Optional[Any] = None,
        sigma_sqr = 0,
        **kwargs
    ):
        """
        Sample x_{t-1} from the model using Euler method.
        
        Args:
            model: The model to sample from.
            x_t: The [N x C x ...] tensor of noisy inputs at time t.
            t: The current timestep.
            t_prev: The previous timestep.
            cond: conditional information.
            **kwargs: Additional arguments for model inference.

        Returns:
            a dict containing the following
            - 'pred_x_prev': x_{t-1}.
            - 'pred_x_0': a prediction of x_0.
        """
        pred_x_0, pred_eps, pred_v = self._get_model_prediction(model, x_t, t, cond, **kwargs)
        pred_x_prev = x_t - (t - math.sqrt((t_prev)**2-sigma_sqr)) * pred_v
        return edict({"pred_x_prev": pred_x_prev, "pred_x_0": pred_x_0})
    
    def sample_once_with_grad(
        self,
        model,
        x_t,
        t: float,
        t_prev: float,
        cond: Optional[Any] = None,
        sigma_sqr=0,
        **kwargs
    ):
        """
        Sample x_{t-1} from the model using Euler method.
        
        Args:
            model: The model to sample from.
            x_t: The [N x C x ...] tensor of noisy inputs at time t.
            t: The current timestep.
            t_prev: The previous timestep.
            cond: conditional information.
            **kwargs: Additional arguments for model inference.

        Returns:
            a dict containing the following
            - 'pred_x_prev': x_{t-1}.
            - 'pred_x_0': a prediction of x_0.
        """
        pred_x_0, pred_eps, pred_v = self._get_model_prediction(model, x_t, t, cond, **kwargs)
        pred_x_prev = x_t - (t - math.sqrt((t_prev)**2-sigma_sqr)) * pred_v
        return edict({"pred_x_prev": pred_x_prev, "pred_x_0": pred_x_0})

    @torch.no_grad()
    def sample(
        self,
        model,
        noise,
        cond: Optional[Any] = None,
        steps: int = 50,
        rescale_t: float = 1.0,
        verbose: bool = True,
        **kwargs
    ):
        """
        Generate samples from the model using Euler method.
        
        Args:
            model: The model to sample from.
            noise: The initial noise tensor.
            cond: conditional information.
            steps: The number of steps to sample.
            rescale_t: The rescale factor for t.
            verbose: If True, show a progress bar.
            **kwargs: Additional arguments for model_inference.

        Returns:
            a dict containing the following
            - 'samples': the model samples.
            - 'pred_x_t': a list of prediction of x_t.
            - 'pred_x_0': a list of prediction of x_0.
        """
        sample = noise
        t_seq = np.linspace(1, 0, steps + 1)
        t_seq = rescale_t * t_seq / (1 + (rescale_t - 1) * t_seq)
        t_pairs = list((t_seq[i], t_seq[i + 1]) for i in range(steps))
        ret = edict({"samples": None, "pred_x_t": [], "pred_x_0": []})
        for t, t_prev in tqdm(t_pairs, desc="Sampling", disable=not verbose):
            out = self.sample_once(model, sample, t, t_prev, cond, **kwargs)
            sample = out.pred_x_prev
            ret.pred_x_t.append(out.pred_x_prev)
            ret.pred_x_0.append(out.pred_x_0)
        ret.samples = sample
        return ret

    @torch.no_grad()
    def sample_noised(
        self,
        model,
        noise,
        cond: Optional[Any] = None,
        steps: int = 50,
        rescale_t: float = 1.0,
        verbose: bool = True,
        **kwargs
    ):
        """
        Generate samples from the model using Euler method.
        
        Args:
            model: The model to sample from.
            noise: The initial noise tensor.
            cond: conditional information.
            steps: The number of steps to sample.
            rescale_t: The rescale factor for t.
            verbose: If True, show a progress bar.
            **kwargs: Additional arguments for model_inference.

        Returns:
            a dict containing the following
            - 'samples': the model samples.
            - 'pred_x_t': a list of prediction of x_t.
            - 'pred_x_0': a list of prediction of x_0.
        """
        sample = noise
        steps = 100
        t_seq = np.linspace(1, 0, steps + 1)
        
        t_seq = rescale_t * t_seq / (1 + (rescale_t - 1) * t_seq)
        t_pairs = list((t_seq[i], t_seq[i + 1]) for i in range(steps))
        ret = edict({"samples": None, "pred_x_t": [], "pred_x_0": []})
        for t, t_prev in tqdm(t_pairs, desc="Sampling", disable=not verbose):
            alpha_t_sqr = 1/(1+t**2)
            alpha_t_prev_sqr = 1/(1+t_prev**2)
            sigma_sqr = (1-alpha_t_prev_sqr)/(1-alpha_t_sqr)*(1-alpha_t_sqr/alpha_t_prev_sqr) /5
            if not torch.is_tensor(sigma_sqr):
                sigma_sqr = sigma_sqr.astype(float)
            out = self.sample_once(model, sample, t, t_prev, cond, sigma_sqr=sigma_sqr, **kwargs)
            sample = out.pred_x_prev
            if t_prev > 0:
                # draw gaussian noise
                eps = torch.randn(sample.shape).to(sample.device)
                sample = sample + sigma_sqr**0.5 * eps

            ret.pred_x_t.append(out.pred_x_prev)
            ret.pred_x_0.append(out.pred_x_0)
        ret.samples = sample
        return ret
    
    @torch.no_grad()
    def sample_noised_with_logprobs(
        self,
        model,
        noise,
        cond: Optional[Any] = None,
        eta = 0.2,
        steps: int = 50,
        rescale_t: float = 1.0,
        verbose: bool = True,
        **kwargs
    ):
        """
        Generate samples from the model using Euler method.
        
        Args:
            model: The model to sample from.
            noise: The initial noise tensor.
            cond: conditional information.
            steps: The number of steps to sample.
            rescale_t: The rescale factor for t.
            verbose: If True, show a progress bar.
            **kwargs: Additional arguments for model_inference.

        Returns:
            a dict containing the following
            - 'samples': the model samples.
            - 'pred_x_t': a list of prediction of x_t.
            - 'pred_x_0': a list of prediction of x_0.
        """
        sample = noise
        t_seq = np.linspace(1, 0, steps + 1)
        
        t_seq = rescale_t * t_seq / (1 + (rescale_t - 1) * t_seq)
        t_pairs = list((t_seq[i], t_seq[i + 1]) for i in range(steps))
        ret = edict({"samples": None, "latents": [sample], "logprobs":[]})
        for t, t_prev in tqdm(t_pairs, desc="Sampling", disable=not verbose):
            alpha_t_sqr = 1/(1+t**2)
            alpha_t_prev_sqr = 1/(1+t_prev**2)
            sigma_sqr = (1-alpha_t_prev_sqr)/(1-alpha_t_sqr)*(1-alpha_t_sqr/alpha_t_prev_sqr) /5
            if not torch.is_tensor(sigma_sqr):
                sigma_sqr = sigma_sqr.astype(float)
            out = self.sample_once(model, sample, t, t_prev, cond, sigma_sqr=sigma_sqr, **kwargs)
            sample = out.pred_x_prev
            if t_prev > 0:
                # draw gaussian noise
                k = sample.feats.view(-1).shape[0]
                eps = torch.randn(sample.feats.shape).to(sample.device)/(k**0.5)
                sample = sample + sigma_sqr**0.5 * eps

                # calculate log probs by using mean=out.pred_x_prev, var=sigma_sqr
                log_prob = (
                    -((sample.detach().feats - out.pred_x_prev.feats) ** 2).sum() / (2 * sigma_sqr)
                    - math.log(sigma_sqr**0.5)
                    #- torch.log(torch.sqrt(2 * torch.as_tensor(math.pi)))*k/2 #it's very small and dominates -  a constant for fixed dimensions
                )
                ret.logprobs.append(log_prob)
            ret.latents.append(sample)
            

        return sample, t_seq, ret["latents"], ret["logprobs"]
    
    def sample_onestep_logprob(
        self,
        model,
        cond,
        x_t,
        x_t_prev,
        t,
        t_prev,
        eta,
        **kwargs):
        # we still use the prior sample trajectory, but now the mean is updated because
        # model is updated!
        # use corresponding ddpm noise schedule
        alpha_t_sqr = 1/(1+t**2)
        alpha_t_prev_sqr = 1/(1+t_prev**2)
        sigma_sqr = (1-alpha_t_prev_sqr)/(1-alpha_t_sqr)*(1-alpha_t_sqr/alpha_t_prev_sqr) * eta
        if not torch.is_tensor(sigma_sqr):
            sigma_sqr = sigma_sqr.astype(float)

        out = self.sample_once_with_grad(model, x_t, t, t_prev, cond, sigma_sqr = sigma_sqr, **kwargs)
        new_pred_mean = out.pred_x_prev

        
        k = x_t_prev.feats.view(-1).shape[0]

        # calculate log probs by using mean=out.pred_x_prev, var=sigma_sqr
        # data is the sparse version, seems like some computation (e.g. spconv)
        # happens with the sparse data and some happen directly on feats

        # so we include both in the log prob (even though the numerical results are
        # identical, to be able to upgrade gradients on both
        diff_data = x_t_prev.detach().data.features - new_pred_mean.data.features
        diff_feats = x_t_prev.detach().feats - new_pred_mean.feats
        diff = (diff_data+diff_feats)/2

        log_prob = (
            -(diff ** 2).sum() / (2 * sigma_sqr)
            - math.log(sigma_sqr**0.5)
            #- torch.log(torch.sqrt(2 * torch.as_tensor(math.pi)))*k/2
        )
        return log_prob


class FlowEulerCfgSampler(ClassifierFreeGuidanceSamplerMixin, FlowEulerSampler):
    """
    Generate samples from a flow-matching model using Euler sampling with classifier-free guidance.
    """
    @torch.no_grad()
    def sample(
        self,
        model,
        noise,
        cond,
        neg_cond,
        steps: int = 50,
        rescale_t: float = 1.0,
        cfg_strength: float = 3.0,
        verbose: bool = True,
        **kwargs
    ):
        """
        Generate samples from the model using Euler method.
        
        Args:
            model: The model to sample from.
            noise: The initial noise tensor.
            cond: conditional information.
            neg_cond: negative conditional information.
            steps: The number of steps to sample.
            rescale_t: The rescale factor for t.
            cfg_strength: The strength of classifier-free guidance.
            verbose: If True, show a progress bar.
            **kwargs: Additional arguments for model_inference.

        Returns:
            a dict containing the following
            - 'samples': the model samples.
            - 'pred_x_t': a list of prediction of x_t.
            - 'pred_x_0': a list of prediction of x_0.
        """
        return super().sample(model, noise, cond, steps, rescale_t, verbose, neg_cond=neg_cond, cfg_strength=cfg_strength, **kwargs)


class FlowEulerGuidanceIntervalSampler(GuidanceIntervalSamplerMixin, FlowEulerSampler):
    """
    Generate samples from a flow-matching model using Euler sampling with classifier-free guidance and interval.
    """
    @torch.no_grad()
    def sample(
        self,
        model,
        noise,
        cond,
        neg_cond,
        steps: int = 50,
        rescale_t: float = 1.0,
        cfg_strength: float = 3.0,
        cfg_interval: Tuple[float, float] = (0.0, 1.0),
        verbose: bool = True,
        **kwargs
    ):
        """
        Generate samples from the model using Euler method.
        
        Args:
            model: The model to sample from.
            noise: The initial noise tensor.
            cond: conditional information.
            neg_cond: negative conditional information.
            steps: The number of steps to sample.
            rescale_t: The rescale factor for t.
            cfg_strength: The strength of classifier-free guidance.
            cfg_interval: The interval for classifier-free guidance.
            verbose: If True, show a progress bar.
            **kwargs: Additional arguments for model_inference.

        Returns:
            a dict containing the following
            - 'samples': the model samples.
            - 'pred_x_t': a list of prediction of x_t.
            - 'pred_x_0': a list of prediction of x_0.
        """
        return super().sample(model, noise, cond, steps, rescale_t, verbose, neg_cond=neg_cond, cfg_strength=cfg_strength, cfg_interval=cfg_interval, **kwargs)
