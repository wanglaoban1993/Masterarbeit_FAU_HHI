import os
import torch
from glob import glob
from ddsm_original import *
from train_sudoku_allinone import *
import argparse

def parse_args():   # to select which presampling .pth should be put into training
    parser = argparse.ArgumentParser("To select automatically the ")

    parser.add_argument('-kw', '--keywords', type=str,
                        help="Number of time steps between <min_time> and <max_time> (default = 400)")
    
    parser.add_argument('-bm', "--boundary_mode", choices=['clamp', 'reflect_boundaries', 'reflection'], default= 'clamp')

    #parser.add_argument('-spd', '--speed_balance', type= choices=['s1', 'sab'], default= 's1')
    parser.add_argument("--speed_balance", type=bool, nargs='?', const=True, default=False,
                        help="Adding speed balance to Jacobi Process (default = False)")
    
    parser.add_argument('-sz', "--batch_size", type= int, default= 256,
                        help="Adding batch size (default = 256)")

    parser.add_argument('-ns', "--num_samples", type= int, default= 200,
                        help="Adding number of samples by sampling during evaluation (default = 200)")
    return parser.parse_args()

def load_model(model_path, model_class, device):
    # 初始化模型架构
    model = model_class(define_relative_encoding())
    
    # 加载模型状态字典
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint)
    
    # 设置模型为评估模式
    model.eval()
    model.to(device)
    
    return model

def evaluate_samples(model, sample_shape, batch_size, device, sampler, num_samples=200, boundary_mode= 'clamp', speed_balanced= False):
    # 使用 Euler_Maruyama_sampler 生成样本; use euler maruyama sampler to generate samples
    samples = sampler(
        model,
        sample_shape,
        batch_size=batch_size,
        max_time=1,
        time_dilation=1,
        num_steps=num_samples,
        random_order=False,
        speed_balanced= speed_balanced,
        #speed_balanced=False,
        device=device,
        boundary_mode= boundary_mode   #boundary_mode='clamp', 'reflect_boundaries', 'reflection'
    )

    # 检查样本是否生成 check if the samples are generated
    if samples is None or len(samples) == 0:
        print("No samples generated.")
        return None
    
    # 评估生成的样本 evaluate the generated samples
    accuracy = sudoku_acc(samples)
    if accuracy is None:
        print("sudoku_acc returned None.")
        return None
    accuracy = sudoku_acc(samples)
    return accuracy

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args = parse_args()
    select_keywords= args.keywords
    select_boundary_mode= args.boundary_mode
    select_speed_balance= args.speed_balance
    select_batch_size= args.batch_size
    select_num_samples= args.num_samples
    print(select_keywords, select_boundary_mode, select_speed_balance, select_batch_size, select_num_samples)

    # 获取父目录路径
    parent_dir = '/home/fe/twang/projects/MA_2024'
    #parent_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))  # 获取父目录路径
    print("Parent directory:", parent_dir)
    
    #model_files_pattern = os.path.join(parent_dir, "score_model_reflection_epoch_*.pth") 
    model_files_pattern = os.path.join(parent_dir, f"{select_keywords}*.pth")  # 构建通配符路径
    print("Model files pattern:", model_files_pattern)
    
    model_files = glob(model_files_pattern)  # 使用通配符列出所有模型文件
    model_files.sort()  # 确保文件按照顺序加载
    print("Matched model files:", model_files)
    
    if not model_files:
        print("No model files found. Please check the directory and file pattern.")
        return
    
    # model_files = glob("score_model_epoch_*.pth")  # 使用通配符列出所有模型文件
    # model_files.sort()  # 确保文件按照顺序加载

    sample_shape = (9, 9, 9)
    #batch_size = 256
    batch_size = select_batch_size
    #num_samples = 200  # 可以根据需要调整
    num_samples = select_num_samples  # 可以根据需要调整

    results = []

    for model_path in model_files:
        print(f"Evaluating model: {model_path}")
        
        # 加载模型
        model = load_model(model_path, ScoreNet, device)
        
        # 生成样本并评估
        accuracy = evaluate_samples(
            model,
            sample_shape,
            batch_size,
            device,
            sampler=Euler_Maruyama_sampler,
            num_samples=num_samples,
            boundary_mode= select_boundary_mode,
            speed_balanced= select_speed_balance

        )
        
        print(f"Sudoku accuracy for {model_path}: {accuracy:.2f}%")
        results.append((model_path, accuracy))
    
    # 输出所有模型的评估结果
    for model_path, accuracy in results:
        print(f"Model: {model_path}, Sudoku accuracy: {accuracy:.2f}%")

if __name__ == "__main__":
    main()

