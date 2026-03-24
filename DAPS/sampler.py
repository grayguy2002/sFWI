import tqdm
import torch
import numpy as np
import torch.nn as nn
from matplotlib import pyplot as plt
from score_sde_pytorch import sampling
from score_sde_pytorch.sampling import (ReverseDiffusionPredictor,
                      LangevinCorrector,
                      EulerMaruyamaPredictor,
                      AncestralSamplingPredictor,
                      NoneCorrector,
                      NonePredictor,
                      AnnealedLangevinDynamics)
import torch.nn.functional as F

def visualize_data(data, mode='velocity', n_samples=None, figsize=(15, 15), save_path=None, title=''):
    """
    统一的可视化函数，可以绘制速度模型或地震记录

    参数:
        data: torch.Tensor, 输入数据
            - 速度模型模式: shape为(batch, channel, height, width)
            - 地震记录模式: shape为(batch, time, receivers)
        mode: str, 可选 'velocity' 或 'seismic'
        n_samples: int, 可选，显示多少个样本
        figsize: tuple, 图像大小
        save_path: str, 可选，保存图像的路径
    """
    plt.figure(figsize=figsize)
    data = data.detach().cpu()

    # 设置要显示的样本数
    if n_samples is None:
        n_samples = len(data)
    n_samples = min(n_samples, len(data))

    # 计算子图布局
    n_rows = int(np.sqrt(n_samples))
    n_cols = int(np.ceil(n_samples / n_rows))

    for i in range(n_samples):
        plt.subplot(n_rows, n_cols, i + 1)

        if mode.lower() == 'velocity':
            # 速度模型可视化
            field = data[i][0].numpy()  # 取第一个通道
            im = plt.imshow(field, cmap='viridis')
            cbar = plt.colorbar(im)
            cbar.set_label('Velocity (m/s)', fontsize=8)  # 减小colorbar标签字体
            cbar.ax.tick_params(labelsize=6)  # 减小colorbar刻度字体
            plt.title(f'Velocity Model {i+1}:{title}', fontsize=8)  # 减小标题字体

        elif mode.lower() == 'seismic':
            # 地震记录可视化
            field = data[i]
            # 使用分位数计算每个记录的颜色范围
            vmin, vmax = torch.quantile(field, torch.tensor([0.05, 0.95]))
            im = plt.imshow(field.T,
                          cmap='gray',
                          vmin=vmin,
                          vmax=vmax)
            cbar = plt.colorbar(im)
            cbar.set_label('Amplitude', fontsize=8)  # 减小colorbar标签字体
            cbar.ax.tick_params(labelsize=6)  # 减小colorbar刻度字体
            plt.title(f'Seismic Record {i+1}', fontsize=8)  # 减小标题字体

        else:
            raise ValueError("mode must be either 'velocity' or 'seismic'")

        plt.axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
    plt.show()

def get_sampler(**kwargs):
    latent = kwargs['latent']
    kwargs.pop('latent')
    if latent:
        raise NotImplementedError
    return DAPS(**kwargs)


class Scheduler(nn.Module):
    """
        Scheduler for diffusion sigma(t) and discretization step size Delta t
    """

    def __init__(self, num_steps=10, sigma_max=100, sigma_min=0.01, sigma_final=None, schedule='linear',
                 timestep='poly-7'):
        """
            Initializes the scheduler with the given parameters.

            Parameters:
                num_steps (int): Number of steps in the schedule.
                sigma_max (float): Maximum value of sigma.
                sigma_min (float): Minimum value of sigma.
                sigma_final (float): Final value of sigma, defaults to sigma_min.
                schedule (str): Type of schedule for sigma ('linear' or 'sqrt').
                timestep (str): Type of timestep function ('log' or 'poly-n').
        """
        super().__init__()
        self.num_steps = num_steps
        self.sigma_max = sigma_max
        self.sigma_min = sigma_min
        self.sigma_final = sigma_final
        if self.sigma_final is None:
            self.sigma_final = self.sigma_min
        self.schedule = schedule
        self.timestep = timestep

        steps = np.linspace(0, 1, num_steps)
        sigma_fn, sigma_derivative_fn, sigma_inv_fn = self.get_sigma_fn(self.schedule)
        time_step_fn = self.get_time_step_fn(self.timestep, self.sigma_max, self.sigma_min)

        time_steps = np.array([time_step_fn(s) for s in steps])
        time_steps = np.append(time_steps, sigma_inv_fn(self.sigma_final))
        sigma_steps = np.array([sigma_fn(t) for t in time_steps])

        # factor = 2\dot\sigma(t)\sigma(t)\Delta t
        factor_steps = np.array(
            [2 * sigma_fn(time_steps[i]) * sigma_derivative_fn(time_steps[i]) * (time_steps[i] - time_steps[i + 1]) for
             i in range(num_steps)])
        self.sigma_steps, self.time_steps, self.factor_steps = sigma_steps, time_steps, factor_steps
        self.factor_steps = [max(f, 0) for f in self.factor_steps]

    def get_sigma_fn(self, schedule):
        """
            Returns the sigma function, its derivative, and its inverse based on the given schedule.
        """
        if schedule == 'sqrt':
            sigma_fn = lambda t: np.sqrt(t)
            sigma_derivative_fn = lambda t: 1 / 2 / np.sqrt(t)
            sigma_inv_fn = lambda sigma: sigma ** 2

        elif schedule == 'linear':
            sigma_fn = lambda t: t
            sigma_derivative_fn = lambda t: 1
            sigma_inv_fn = lambda t: t
        else:
            raise NotImplementedError
        return sigma_fn, sigma_derivative_fn, sigma_inv_fn

    def get_time_step_fn(self, timestep, sigma_max, sigma_min):
        """
            Returns the time step function based on the given timestep type.
        """
        if timestep == 'log':
            get_time_step_fn = lambda r: sigma_max ** 2 * (sigma_min ** 2 / sigma_max ** 2) ** r
        elif timestep.startswith('poly'):
            p = int(timestep.split('-')[1])
            get_time_step_fn = lambda r: (sigma_max ** (1 / p) + r * (sigma_min ** (1 / p) - sigma_max ** (1 / p))) ** p
        else:
            raise NotImplementedError
        return get_time_step_fn


class DiffusionSampler(nn.Module):
    """
        Diffusion sampler for reverse SDE or PF-ODE
    """

    def __init__(self, scheduler, solver='euler'):
        """
            Initializes the diffusion sampler with the given scheduler and solver.

            Parameters:
                scheduler (Scheduler): Scheduler instance for managing sigma and timesteps.
                solver (str): Solver method ('euler').
        """
        super().__init__()
        self.scheduler = scheduler
        self.solver = solver

    def sample(self, model, x_start, SDE=False, record=False, verbose=False):
        """
            Samples from the diffusion process using the specified model.

            Parameters:
                model (DiffusionModel): Diffusion model supports 'score' and 'tweedie'
                x_start (torch.Tensor): Initial state.
                SDE (bool): Whether to use Stochastic Differential Equations.
                record (bool): Whether to record the trajectory.
                verbose (bool): Whether to display progress bar.

            Returns:
                torch.Tensor: The final sampled state.
        """
        if self.solver == 'euler':
            return self._euler(model, x_start, SDE, record, verbose)
        else:
            raise NotImplementedError

    def _euler(self, model, x_start, SDE=False, record=False, verbose=False):
        """
            Euler's method for sampling from the diffusion process.
        """
        if record:
            self.trajectory = Trajectory()
        pbar = tqdm.trange(self.scheduler.num_steps) if verbose else range(self.scheduler.num_steps)

        x = x_start
        for step in pbar:
            sigma, factor = self.scheduler.sigma_steps[step], self.scheduler.factor_steps[step]
            score = model.score(x, sigma)
            if SDE:
                epsilon = torch.randn_like(x)
                x = x + factor * score + np.sqrt(factor) * epsilon
            else:
                x = x + factor * score * 0.5
            # record
            if record:
                if SDE:
                    self._record(x, score, sigma, factor, epsilon)
                else:
                    self._record(x, score, sigma, factor)
        return x

    def _record(self, x, score, sigma, factor, epsilon=None):
        """
            Records the intermediate states during sampling.
        """
        self.trajectory.add_tensor(f'xt', x)
        self.trajectory.add_tensor(f'score', score)
        self.trajectory.add_value(f'sigma', sigma)
        self.trajectory.add_value(f'factor', factor)
        if epsilon is not None:
            self.trajectory.add_tensor(f'epsilon', epsilon)

    def get_start(self, ref):
        """
            Generates a random initial state based on the reference tensor.

            Parameters:
                ref (torch.Tensor): Reference tensor for shape and device.

            Returns:
                torch.Tensor: Initial random state.
        """
        x_start = torch.randn_like(ref) * self.scheduler.sigma_max
        return x_start


class LatentDiffusionSampler(DiffusionSampler):
    """
        Latent Diffusion sampler for reverse SDE or PF-ODE
    """

    def __init__(self, scheduler, solver='euler'):
        """
            Initializes the latent diffusion sampler with the given scheduler and solver.

            Parameters:
                scheduler (Scheduler): Scheduler instance for managing sigma and timesteps.
                solver (str): Solver method ('euler').
        """
        super().__init__(scheduler, solver)

    def sample(self, model, z_start, SDE=False, record=False, verbose=False, return_latent=True):
        """
            Samples from the latent diffusion process using the specified model.

            Parameters:
                model (LatentDiffusionModel): Diffusion model supports 'score', 'tweedie', 'encode' and 'decode'
                z_start (torch.Tensor): Initial latent state.
                SDE (bool): Whether to use Stochastic Differential Equations.
                record (bool): Whether to record the trajectory.
                verbose (bool): Whether to display progress bar.
                return_latent (bool): Whether to return the latent state or decoded state.

            Returns:
                torch.Tensor: The final sampled state (latent or decoded).
        """
        if self.solver == 'euler':
            z0 = self._euler(model, z_start, SDE, record, verbose)
        else:
            raise NotImplementedError
        if return_latent:
            return z0
        else:
            x0 = model.decode(z0)
            return x0


class LangevinDynamics(nn.Module):
    """
        Langevin Dynamics sampling method.
    """

    def __init__(self, num_steps, lr, tau=0.01, lr_min_ratio=0.01,
                 lambda_prior=1.0, lambda_prior_min_ratio=1.0):
        """
            Initializes the Langevin dynamics sampler with the given parameters.

            Parameters:
                num_steps (int): Number of steps in the sampling process.
                lr (float): Learning rate.
                tau (float): Noise parameter.
                lr_min_ratio (float): Minimum learning rate ratio.
                lambda_prior (float): Prior term scale on ||x - x0hat||^2.
                lambda_prior_min_ratio (float): Min ratio for lambda_prior at early annealing steps.
        """
        super().__init__()
        self.num_steps = num_steps
        self.lr = lr
        self.tau = tau
        self.lr_min_ratio = lr_min_ratio
        self.lambda_prior = lambda_prior
        self.lambda_prior_min_ratio = lambda_prior_min_ratio

    def sample(self, x0hat, operator, measurement, sigma, ratio, record=False, verbose=False):
        """
            Samples using Langevin dynamics.

            Parameters:
                x0hat (torch.Tensor): Initial state.
                operator (Operator): Operator module.
                measurement (torch.Tensor): Measurement tensor.
                sigma (float): Current sigma value.
                ratio (float): Current step ratio.
                record (bool): Whether to record the trajectory.
                verbose (bool): Whether to display progress bar.

            Returns:
                torch.Tensor: The final sampled state.
        """
        if record:
            self.trajectory = Trajectory()
        pbar = tqdm.trange(self.num_steps) if verbose else range(self.num_steps)
        lr = self.get_lr(ratio)
        lambda_prior = self.get_lambda_prior(ratio)
        x = x0hat.clone().detach().requires_grad_(True)
        optimizer = torch.optim.SGD([x], lr)
        for _ in pbar:
            optimizer.zero_grad()
            sigma_eff = max(float(sigma), 1e-8)
            # data term: compare forward response F(x) with measurement once (avoid F(F(x)))
            loss = operator.error(x, measurement).sum() / (2 * self.tau ** 2)
            loss += lambda_prior * ((x - x0hat.detach()) ** 2).sum() / (2 * sigma_eff ** 2)
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                epsilon = torch.randn_like(x)
                x.data = x.data + np.sqrt(2 * lr) * epsilon

            # early stopping with NaN
            if torch.isnan(x).any():
                return torch.zeros_like(x)

            # record
            if record:
                self._record(x, epsilon, loss)
        return x.detach()

    def _record(self, x, epsilon, loss):
        """
            Records the intermediate states during sampling.
        """
        self.trajectory.add_tensor(f'xi', x)
        self.trajectory.add_tensor(f'epsilon', epsilon)
        self.trajectory.add_value(f'loss', loss)

    def get_lr(self, ratio):
        """
            Computes the learning rate based on the given ratio.
        """
        p = 1
        multiplier = (1 ** (1 / p) + ratio * (self.lr_min_ratio ** (1 / p) - 1 ** (1 / p))) ** p
        return multiplier * self.lr

    def get_lambda_prior(self, ratio):
        """
            Computes lambda_prior based on annealing ratio.
            Early stage uses lower prior (more exploration), later stage increases prior (more convergence).
        """
        r = min(max(float(ratio), 0.0), 1.0)
        min_r = min(max(float(self.lambda_prior_min_ratio), 0.0), 1.0)
        multiplier = min_r + r * (1.0 - min_r)
        return self.lambda_prior * multiplier


class DAPS(nn.Module):
    def __init__(self, annealing_scheduler_config, diffusion_scheduler_config, lgvd_config,
                 sde, inverse_scaler, sampling_eps=1e-5, w_threshold=0.8, max_attempts=1):
        super().__init__()
        annealing_scheduler_config, diffusion_scheduler_config = self._check(annealing_scheduler_config,
                                                                           diffusion_scheduler_config)
        self.annealing_scheduler = Scheduler(**annealing_scheduler_config)
        self.diffusion_scheduler_config = diffusion_scheduler_config
        self.lgvd = LangevinDynamics(**lgvd_config)
        self.w_threshold = w_threshold
        self.max_attempts = max_attempts
        # 保存PC sampling所需的参数
        self.sde = sde
        self.inverse_scaler = inverse_scaler
        self.sampling_eps = sampling_eps

    def get_initial_sample(self, model, x_start, measurement, verbose=False, evaluator_us=None, seed=None):
        """
        Random Score Search (RSS): 使用PC sampling获取满足Wasserstein距离要求的初始样本
        注意： RSS算法对于seed的要求是None
        """
        best_x0hat = None
        best_w_dist = float('inf')

        # PC sampling参数设置
        shape = x_start.shape
        predictor = ReverseDiffusionPredictor
        corrector = LangevinCorrector
        snr = 0.16
        n_steps = 1
        probability_flow = False
        # 设置PC sampler, 这里没有调用sampling.get_sampling_fn(),也即没有使用config当中指定的~ors
        sampling_fn = sampling.get_pc_sampler( #reminder: sampling is .py file
            sde=self.sde,
            shape=shape,
            predictor=predictor,
            corrector=corrector,
            inverse_scaler=self.inverse_scaler,
            snr=snr,
            n_steps=n_steps,
            probability_flow=probability_flow,
            continuous=True,  # 假设使用连续时间
            eps=self.sampling_eps,
            device=x_start.device,
            seed=seed # need to be changed if max_attempts is not 1!
        )

        for attempt in range(self.max_attempts):
            # 使用PC sampling
            x0hat, n = sampling_fn(model)

            x0hat_high = F.interpolate(
            x0hat,
            size=(128,128), #hard code here
            mode='bilinear',
            align_corners=True
        )
            # 计算Wasserstein距离
            with torch.no_grad():
                w_dist = evaluator_us(None, measurement, x0hat_high)
            if verbose:
                print(f"Attempt {attempt + 1}, Wasserstein distance: {w_dist['w2dist_unsupervised'].item()}")

            # 更新最佳结果
            if w_dist['w2dist_unsupervised'].item() < best_w_dist:
                best_w_dist = w_dist['w2dist_unsupervised'].item()
                best_x0hat = x0hat.clone()

            # 如果距离小于阈值，直接返回
            if w_dist['w2dist_unsupervised'].item() < self.w_threshold:
                if verbose:
                    print(f"Found satisfactory sample with W-distance: {w_dist['w2dist_unsupervised'].item()}")
                return x0hat

        if verbose:
            print(f"Max attempts reached. Using best sample with W-distance: {best_w_dist}")
        return best_x0hat

    def get_SM_entry(self, model, x_start, measurement, verbose=False, evaluator_us=None, seed=None):
        """
        calculate one entry of the Similarity Matrix.
        """
        # PC sampling参数设置
        shape = x_start.shape
        predictor = ReverseDiffusionPredictor
        corrector = LangevinCorrector
        snr = 0.16
        n_steps = 1 # 一步采样
        probability_flow = False

        # 设置PC sampler
        sampling_fn = sampling.get_pc_sampler(
            sde=self.sde,
            shape=shape,
            predictor=predictor,
            corrector=corrector,
            inverse_scaler=self.inverse_scaler,
            snr=snr,
            n_steps=n_steps,
            probability_flow=probability_flow,
            continuous=True,
            eps=self.sampling_eps,
            device=x_start.device,
            seed=seed
        )

        # 使用PC sampling
        x0hat, n = sampling_fn(model)
        x0hat_high = F.interpolate(
            x0hat,
            size=(128,128),
            mode='bilinear',
            align_corners=True
        )

        # 计算Wasserstein距离
        with torch.no_grad():
            w_dist = evaluator_us(None, measurement, x0hat_high)

        if verbose:
            print(f"Wasserstein distance: {w_dist['w2dist_unsupervised'].item()}")

        return w_dist['w2dist_unsupervised'].item() # float


    def explicit_sample(self, model, x_start, operator, measurement, evaluator_us=None, evaluator=None, record=False, verbose=False, seed=None, **kwargs):
        '''
        根据显式的seed来采样
        x_start： capture the tensor device to run on
        '''
        if record:
            self.trajectory = Trajectory()

        # 首先获取满足要求的初始样本
        x_start = self.get_initial_sample(model, x_start, measurement, verbose, evaluator_us, seed)

        pbar = tqdm.trange(self.annealing_scheduler.num_steps) if verbose else range(self.annealing_scheduler.num_steps)

        xt = x_start
        if verbose:
            visualize_data(xt.detach().cpu(), title='Initial xt')

        for step in pbar:
            sigma = self.annealing_scheduler.sigma_steps[step]
            # 1. reverse diffusion
            diffusion_scheduler = Scheduler(**self.diffusion_scheduler_config, sigma_max=sigma)
            sampler = DiffusionSampler(diffusion_scheduler)
            x0hat = sampler.sample(model, xt, SDE=False, verbose=False)
            if step % 10 == 0 and verbose:
              visualize_data(x0hat,title='x0hat')

            # 2. langevin dynamics
            x0y = self.lgvd.sample(x0hat, operator, measurement, sigma, step / self.annealing_scheduler.num_steps)
            if step % 10 == 0 and verbose:
              visualize_data(x0y,title='x0y')
            # 3. forward diffusion
            xt = x0y + torch.randn_like(x0y) * self.annealing_scheduler.sigma_steps[step + 1]
            if step % 10 == 0 and verbose:
              visualize_data(xt,title='xt')
            # 4. evaluation
            x0hat_results = x0y_results = {}
            if evaluator and 'gt' in kwargs:
                with torch.no_grad():
                    gt = kwargs['gt']
                    x0hat_results = evaluator(gt, measurement, x0hat)
                    x0y_results = evaluator(gt, measurement, x0y)

                # record
                if verbose:
                    main_eval_fn_name = evaluator.main_eval_fn_name
                    pbar.set_postfix({
                        'x0hat' + '_' + main_eval_fn_name: f"{x0hat_results[main_eval_fn_name].item():.2f}",
                        'x0y' + '_' + main_eval_fn_name: f"{x0y_results[main_eval_fn_name].item():.2f}",
                    })
            if record:
                self._record(xt, x0y, x0hat, sigma, x0hat_results, x0y_results)
        return x0hat

    def sample(self, model, x_start, operator, measurement, evaluator_us=None, evaluator=None, record=False, verbose=False, seed=None, return_batch=False, **kwargs):
        '''
        根据指定的batch_size一次性生成所有采样点，并计算它们与给定measurement的Wasserstein距离。
        此方法取代了原有的循环结构，以实现批处理加速。

        Args:
            model: score-based model.
            x_start (torch.Tensor): 一个用于获取shape和device的参考tensor, shape: [1, C, H, W].
            operator: The forward operator.
            measurement (torch.Tensor): 单个ground truth对应的测量数据.
            evaluator_us: 用于计算Wasserstein距离的评估器.
            seed (range or list): 用于决定批处理大小(batch_size), 例如 range(50).
                                  注意：此模式下将生成一个随机批次，单个seed的轨迹不再固定。
                                  如需复现，请在外部设置torch.manual_seed().
        Returns:
            torch.Tensor: 一个包含批次中每个样本与measurement之间Wasserstein距离的tensor.
        '''
        # 1. 根据 seed 参数的长度决定 batch_size
        try:
            batch_size = len(seed)
        except TypeError:
            # 如果seed不是集合类型（如单个整数），则无法确定批次大小
            raise ValueError("In batch mode, 'seed' must be a collection (e.g., range or list) to determine the batch size.")

        if verbose:
            print(f"Starting batched Group Score Search with batch_size = {batch_size}")

        # 2. 设置PC Sampler的参数, 特别是shape
        _, C, H, W = x_start.shape
        shape = (batch_size, C, H, W)
        predictor = ReverseDiffusionPredictor
        corrector = LangevinCorrector
        snr = 0.16
        n_steps = 1
        probability_flow = False

        # 3. 创建PC Sampler实例
        # 注意: 内部seed设为None，以生成一个随机批次
        sampling_fn = sampling.get_pc_sampler(
            sde=self.sde,
            shape=shape,
            predictor=predictor,
            corrector=corrector,
            inverse_scaler=self.inverse_scaler,
            snr=snr,
            n_steps=n_steps,
            probability_flow=probability_flow,
            continuous=True,
            eps=self.sampling_eps,
            device=x_start.device,
            seed=None
        )

        # 4. 一次性生成一个批次的样本
        x0hat_batch, n = sampling_fn(model)

        # 5. 对生成的批次进行上采样 (如果需要)
        x0hat_high_batch = F.interpolate(
            x0hat_batch,
            size=(128, 128),  # 保持与原有逻辑一致
            mode='bilinear',
            align_corners=True
        )

        # 6. 批量计算Wasserstein距离
        with torch.no_grad():
            # 假设evaluator_us可以处理批次化的x0hat_high_batch
            w_dists_dict = evaluator_us(None, measurement, x0hat_high_batch)

        # 提取距离tensor
        w_distances = w_dists_dict['w2dist_unsupervised']

        if verbose:
            print(f"Computed Wasserstein distances for the batch. Shape: {w_distances.shape}")

        # 7. 根据 return_batch 参数决定返回值
        if return_batch:
            return w_distances, x0hat_batch
        else:
            return w_distances

    def _record(self, xt, x0y, x0hat, sigma, x0hat_results, x0y_results):
        """
            Records the intermediate states during sampling.
        """
        self.trajectory.add_tensor(f'xt', xt)
        self.trajectory.add_tensor(f'x0y', x0y)
        self.trajectory.add_tensor(f'x0hat', x0hat)
        self.trajectory.add_value(f'sigma', sigma)
        for name in x0hat_results.keys():
            self.trajectory.add_value(f'x0hat_{name}', x0hat_results[name])
        for name in x0y_results.keys():
            self.trajectory.add_value(f'x0y_{name}', x0y_results[name])

    def _check(self, annealing_scheduler_config, diffusion_scheduler_config):
        """
            Checks and updates the configurations for the schedulers.
        """
        # sigma_max of diffusion scheduler change each step
        if 'sigma_max' in diffusion_scheduler_config:
            diffusion_scheduler_config.pop('sigma_max')

        # Keep caller-provided sigma_final if present; fallback to legacy default 0.
        if 'sigma_final' not in annealing_scheduler_config or annealing_scheduler_config['sigma_final'] is None:
            annealing_scheduler_config['sigma_final'] = 0
        return annealing_scheduler_config, diffusion_scheduler_config

    def get_start(self, ref):
        """
            Generates a random initial state based on the reference tensor.

            Parameters:
                ref (torch.Tensor): Reference tensor for shape and device.

            Returns:
                torch.Tensor: Initial random state.
        """
        x_start = torch.randn_like(ref) * self.annealing_scheduler.sigma_max
        return x_start


class Trajectory(nn.Module):
    """
        Class for recording and storing trajectory data.
    """

    def __init__(self):
        super().__init__()
        self.tensor_data = {}
        self.value_data = {}
        self._compile = False

    def add_tensor(self, name, images):
        """
            Adds image data to the trajectory.

            Parameters:
                name (str): Name of the image data.
                images (torch.Tensor): Image tensor to add.
        """
        if name not in self.tensor_data:
            self.tensor_data[name] = []
        self.tensor_data[name].append(images.detach().cpu())

    def add_value(self, name, values):
        """
            Adds value data to the trajectory.

            Parameters:
                name (str): Name of the value data.
                values (any): Value to add.
        """
        if name not in self.value_data:
            self.value_data[name] = []
        self.value_data[name].append(values)

    def compile(self):
        """
            Compiles the recorded data into tensors.

            Returns:
                Trajectory: The compiled trajectory object.
        """
        if not self._compile:
            self._compile = True
            for name in self.tensor_data.keys():
                self.tensor_data[name] = torch.stack(self.tensor_data[name], dim=0)
            for name in self.value_data.keys():
                self.value_data[name] = torch.tensor(self.value_data[name])
        return self

    @classmethod
    def merge(cls, trajs):
        """
            Merge a list of compiled trajectories from different batches

            Returns:
                Trajectory: The merged and compiled trajectory object.
        """
        merged_traj = cls()
        for name in trajs[0].tensor_data.keys():
            merged_traj.tensor_data[name] = torch.cat([traj.tensor_data[name] for traj in trajs], dim=1)
        for name in trajs[0].value_data.keys():
            merged_traj.value_data[name] = trajs[0].value_data[name]
        return merged_traj
