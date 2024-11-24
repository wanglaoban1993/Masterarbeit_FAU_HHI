import argparse
import os.path

import torch
import matplotlib.pyplot as plt

from ddsm import noise_factory

def parse_args():
    parser = argparse.ArgumentParser("Pre generate jacobi process values with specified number of "
                                     "categories and time points")

    parser.add_argument("-n", "--num_samples", type=int,
                        help="Number of the different samples pre generated (default = 100000)",
                        default=100000) 
    parser.add_argument("-c", "--num_cat", type=int,
                        help="Number of categories", required=True) 
    parser.add_argument("-t", '--num_time_steps', type=int,
                        help="Number of time steps between <min_time> and <max_time> (default = 400)",
                        default=400) 
    parser.add_argument("--speed_balance", type=bool, nargs='?', const=True, default=False,
                        help="Adding speed balance to Jacobi Process (default = False)")
    # parser.add_argument("--speed_balance", action='store_true',
    #                     help="Adding speed balance to Jacobi Process")
    parser.add_argument("--max_time", type=float,
                        help="Last time point (default = 4.0)",
                        default=4.0)
    parser.add_argument("--out_path", type=str,
                        help="Path to output directory, where precomputed noise will be saved",
                        default=".")
    parser.add_argument("--order", type=int,
                        help="Order of Jacobi polynomials. It affects precision of the noise overall (default = 1000)",
                        default=1000)
    parser.add_argument("--steps_per_tick", type=int,
                        help="Number of steps per time tick. One tick is (<max_time> - <min_time>) / num_time_steps "
                             "(default = 200)",
                        default=200)
    parser.add_argument("--mode", choices=['path', 'independent'],
                        help="Mode for calculating values at each time points. If it is path, previous time point "
                             "will be chosen. If it is independent, each time point will be computed from <min_time>.",
                        default='path')
    parser.add_argument("--logspace", action='store_true',
                        help="Use logspace time points")
    parser.add_argument("--boundary_mode", choices=['clamp', 'reflect_boundaries', 'reflection'], default= 'clamp')
    return parser.parse_args()


#################################### Visulasition #################################
def visualize_diffusion(timepoints, samples, plot_name, title="Diffusion Process", num_samples_to_plot=1000):
    # """
    # Visualize the diffusion process at different time points.
    
    # Parameters:
    # - timepoints: A tensor containing the time points.
    # - samples: A tensor containing the samples at different time points. 
    # - title: Title of the plot.
    # - num_samples_to_plot: Number of samples to plot for visualization.
    # """
    # Ensure we don't try to plot more samples than we have
    num_samples_to_plot = min(num_samples_to_plot, samples.shape[0])
    
    plt.figure(figsize=(12, 8))
    for i, t in enumerate(timepoints):
        # Assuming samples are 2-dimensional for visualization
        plt.scatter(samples[:num_samples_to_plot, i, 0], samples[:num_samples_to_plot, i, 1], s=1, alpha=0.6, label=f"t={t:.2f}")
    
    plt.title(title)
    plt.xlabel("Time(t)")
    plt.ylabel("Xt")
    plt.legend()
    plt.savefig(f'sudoku/{plot_name}.png')
    plt.show()

if __name__ == '__main__':
    args = parse_args()

    if not os.path.exists(args.out_path):
        os.makedirs(args.out_path)
    elif not os.path.isdir(args.out_path):
        print(f"{args.out_path} is already exists and it is not a directory")
        exit(1)

    #str_speed = ".speed_balance" if args.speed_balance else ""
    if args.speed_balance is True:
       s_name= 'sab'  
    else:
       s_name= 's1'
    print('s_name in presample_noise.py:', s_name)

    boundary_mode = None
    chosen_mode = args.boundary_mode
    valid_choices = ["clamp", "reflect_boundaries", "reflection"]
    if chosen_mode in valid_choices:
        boundary_mode = chosen_mode
    
    filename = f'steps{args.num_time_steps}.cat{args.num_cat}.time{args.max_time}.' \
               f'samples{args.num_samples}.{boundary_mode}.{s_name}'
    #filepath = os.path.join(args.out_path, filename +"_reflect_boundaries"+ "_s1"+ ".pth")
    #filepath = os.path.join(args.out_path, filename + "_reflection"+ "_s1"+ ".pth")
    #filepath = os.path.join(args.out_path, filename + "_independent_model"+ "_s1"+ ".pth")
    #filepath = os.path.join(args.out_path, filename + "_path_model"+ "_s1"+ ".pth")

    #filepath = os.path.join(args.out_path, filename +"_reflect_boundaries"+ "_sab"+ ".pth")
    #filepath = os.path.join(args.out_path, filename + "_reflection"+ "_sab"+ ".pth")
    #filepath = os.path.join(args.out_path, filename + "_independent_model"+ "_sab"+ ".pth")
    #filepath = os.path.join(args.out_path, filename + "_path_model"+ "_sab"+ ".pth")

    filepath = os.path.join(args.out_path, filename + ".pth")

    if os.path.exists(filepath):
        print("File is already exists.")
        exit(1)

    torch.set_default_dtype(torch.float64)
    
    device="cuda"
    alpha = torch.ones(args.num_cat - 1)
    beta =  torch.arange(args.num_cat - 1, 0, -1)
    #alpha = torch.ones(args.num_cat - 1).to(device) * 2  # 所有元素都是2
    #beta = torch.ones(args.num_cat - 1).to(device) * 2  # 所有元素都是2
    # alpha = torch.arange(1, args.num_cat).to(device)  # 从1递增到 args.num_cat-1
    # beta = torch.arange(args.num_cat - 1, 0, -1).to(device)  # 从 args.num_cat-1 递减到1
    # alpha = torch.rand(args.num_cat - 1).to(device) * 2  # 随机值在 [0, 2] 之间
    # beta = torch.rand(args.num_cat - 1).to(device) * 2  # 随机值在 [0, 2] 之间




    v_one, v_zero, v_one_loggrad, v_zero_loggrad, timepoints = noise_factory(args.num_samples,
                                                                             args.num_time_steps,
                                                                             alpha,
                                                                             beta,
                                                                             total_time=args.max_time,
                                                                             order=args.order,
                                                                             time_steps=args.steps_per_tick,
                                                                             logspace=args.logspace,
                                                                             speed_balanced=args.speed_balance,
                                                                             mode=args.mode,
                                                                             boundary_mode= args.boundary_mode)

    # v_one, v_zero, timepoints = noise_factory(args.num_samples,
    #                                                                          args.num_time_steps,
    #                                                                          alpha,
    #                                                                          beta,
    #                                                                          total_time=args.max_time,
    #                                                                          order=args.order,
    #                                                                          time_steps=args.steps_per_tick,
    #                                                                          logspace=args.logspace,
    #                                                                          speed_balanced=args.speed_balance,
    #                                                                          mode=args.mode,
    #                                                                          boundary_mode= args.boundary_mode)

    v_one = v_one.cpu()
    v_zero = v_zero.cpu()
    v_one_loggrad = v_one_loggrad.cpu()
    v_zero_loggrad = v_zero_loggrad.cpu()
    timepoints = torch.FloatTensor(timepoints)

    # 假设 timepoints 和 v_one 已经生成
    plot_name= filename.replace(".", "_")
    visualize_diffusion(timepoints, v_one, plot_name, title="Jacobi Diffusion Process", num_samples_to_plot=1000)

    torch.save((v_one, v_zero, v_one_loggrad, v_zero_loggrad, timepoints), filepath)
    #torch.save((v_one, v_zero, timepoints), filepath)
