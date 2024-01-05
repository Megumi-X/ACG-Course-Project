import time
import torch
def gpu_acc_experiments():

    result_dict = dict()
    from ball_on_pin import simulatorController
    print("Start cpu experiments for ballOnPin!")
    simulatorController.cpu()
    TOTAL_TIME=120*10*0.001
    NORMAL_SIM_STEP=0.01 # collision stm step would be this devided by 10.
    RENDER_STEP=0.002

    frame_num=0
    current_time=0.0
    previous_frame=-1

    start_time = time.time()
    while current_time < TOTAL_TIME:
        current_time += simulatorController.Forward(NORMAL_SIM_STEP,zoomin_factor_for_collision=2.)
        current_frame = int(current_time / RENDER_STEP)
        if current_frame - previous_frame >= 1:
            print(f"{current_time}/{TOTAL_TIME}")
            position_np = simulator.position.detach().cpu().numpy()
            previous_frame = current_frame
            np.save(folder / "{:04d}.npy".format(current_frame + 1), position_np)
    end_time = time.time()
    result_dict['cpu_ballOnPin'] = end_time - start_time
    
    print("Start gpu experiments for ballOnPin!")
    from ball_on_pin import simulatorController
    simulatorController.cuda()
    start_time = time.time()
    while current_time < TOTAL_TIME:
        current_time += simulatorController.Forward(NORMAL_SIM_STEP,zoomin_factor_for_collision=2.)
        current_frame = int(current_time / RENDER_STEP)
        if current_frame - previous_frame >= 1:
            print(f"{current_time}/{TOTAL_TIME}")
            position_np = simulator.position.detach().cpu().numpy()
            previous_frame = current_frame
            np.save(folder / "{:04d}.npy".format(current_frame + 1), position_np)
    end_time = time.time()
    result_dict['gpu_ballOnPin'] = end_time - start_time





    from bunny_in_pipe import simulatorController
    print("Start cpu experiments for bunnyInPipe!")
    simulatorController.cpu()
    TOTAL_TIME = 200*10*0.001
    NORMAL_SIM_STEP = 0.01
    RENDER_STEP = 0.01

    current_time = 0.0
    previous_frame = -1
    start_time = time.time()
    while current_time < TOTAL_TIME:
        current_time += simulatorController.Forward(NORMAL_SIM_STEP,zoomin_factor_for_collision=2.)
        current_frame = int(current_time/RENDER_STEP)
        if current_frame - previous_frame >= 1:
            position_np = simulator.position.detach().cpu().numpy()
            np.save(folder / "{:04d}.npy".format(current_frame + 1), position_np)
            previous_frame = current_frame
    end_time = time.time()
    result_dict['cpu_bunnyInPipe'] = end_time - start_time

    from bunny_in_pipe import simulatorController
    simulatorController.cuda()
    TOTAL_TIME = 200*10*0.001
    NORMAL_SIM_STEP = 0.01
    RENDER_STEP = 0.01

    current_time = 0.0
    previous_frame = -1
    start_time = time.time()
    while current_time < TOTAL_TIME:
        current_time += simulatorController.Forward(NORMAL_SIM_STEP,zoomin_factor_for_collision=2.)
        current_frame = int(current_time/RENDER_STEP)
        if current_frame - previous_frame >= 1:
            position_np = simulator.position.detach().cpu().numpy()
            np.save(folder / "{:04d}.npy".format(current_frame + 1), position_np)
            previous_frame = current_frame
    end_time = time.time()
    result_dict['gpu_bunnyInPipe'] = end_time - start_time




    from bunny_on_pins import simulatorController
    simulatorController.cpu()
    TOTAL_TIME = 600*10*0.001
    NORMAL_SIM_STEP = 0.01
    RENDER_STEP = 0.01

    current_time = 0.0
    previous_frame = -1
    start_time = time.time()
    while current_time < TOTAL_TIME:
        current_time += simulatorController.Forward(NORMAL_SIM_STEP,zoomin_factor_for_collision=2.)
        current_frame = int(current_time/RENDER_STEP)
        if current_frame - previous_frame >= 1:
            position_np = simulator.position.detach().cpu().numpy()
            np.save(folder / "{:04d}.npy".format(current_frame + 1), position_np)
            previous_frame = current_frame
    end_time = time.time()
    result_dict['cpu_bunnyOnPins'] = end_time - start_time

    from bunny_on_pins import simulatorController
    simulatorController.cuda()
    TOTAL_TIME = 600*10*0.001
    NORMAL_SIM_STEP = 0.01
    RENDER_STEP = 0.01

    current_time = 0.0
    previous_frame = -1
    start_time = time.time()
    while current_time < TOTAL_TIME:
        current_time += simulatorController.Forward(NORMAL_SIM_STEP,zoomin_factor_for_collision=2.)
        current_frame = int(current_time/RENDER_STEP)
        if current_frame - previous_frame >= 1:
            position_np = simulator.position.detach().cpu().numpy()
            np.save(folder / "{:04d}.npy".format(current_frame + 1), position_np)
            previous_frame = current_frame
    end_time = time.time()
    result_dict['gpu_bunnyOnPins'] = end_time - start_time

    return result_dict

def motionpro_acc_experiments():

    result_dict = dict()
    from ball_on_pin import simulatorController
    print("Start MotionPro experiments for ballOnPin!")
    simulatorController.cuda()
    TOTAL_TIME=120*10*0.001
    NORMAL_SIM_STEP=0.01 # collision stm step would be this devided by 10.
    RENDER_STEP=0.002

    frame_num=0
    current_time=0.0
    previous_frame=-1

    start_time = time.time()
    while current_time < TOTAL_TIME:
        current_time += simulatorController.Forward(NORMAL_SIM_STEP,zoomin_factor_for_collision=2.)
        current_frame = int(current_time / RENDER_STEP)
        if current_frame - previous_frame >= 1:
            print(f"{current_time}/{TOTAL_TIME}")
            position_np = simulator.position.detach().cpu().numpy()
            previous_frame = current_frame
            np.save(folder / "{:04d}.npy".format(current_frame + 1), position_np)
    end_time = time.time()
    result_dict['MP_ballOnPin_2'] = end_time - start_time
    

    NORMAL_SIM_STEP=0.001
    print("Start normal experiments for ballOnPin!")
    from ball_on_pin import simulatorController
    simulatorController.cuda()
    start_time = time.time()
    while current_time < TOTAL_TIME:
        current_time += simulatorController.Forward(NORMAL_SIM_STEP,zoomin_factor_for_collision=1.)
        current_frame = int(current_time / RENDER_STEP)
        if current_frame - previous_frame >= 1:
            print(f"{current_time}/{TOTAL_TIME}")
            position_np = simulator.position.detach().cpu().numpy()
            previous_frame = current_frame
            np.save(folder / "{:04d}.npy".format(current_frame + 1), position_np)
    end_time = time.time()
    result_dict['Normal_ballOnPin'] = end_time - start_time





    from bunny_in_pipe import simulatorController
    print("Start MotionPro experiments for bunnyInPipe!")
    simulatorController.cuda()
    TOTAL_TIME = 200*10*0.001
    NORMAL_SIM_STEP = 0.01
    RENDER_STEP = 0.01

    current_time = 0.0
    previous_frame = -1
    start_time = time.time()
    while current_time < TOTAL_TIME:
        current_time += simulatorController.Forward(NORMAL_SIM_STEP,zoomin_factor_for_collision=2.)
        current_frame = int(current_time/RENDER_STEP)
        if current_frame - previous_frame >= 1:
            position_np = simulator.position.detach().cpu().numpy()
            np.save(folder / "{:04d}.npy".format(current_frame + 1), position_np)
            previous_frame = current_frame
    end_time = time.time()
    result_dict['MP_bunnyInPipe_2'] = end_time - start_time

    from bunny_in_pipe import simulatorController
    simulatorController.cuda()
    TOTAL_TIME = 200*10*0.001
    NORMAL_SIM_STEP = 0.01
    RENDER_STEP = 0.005

    current_time = 0.0
    previous_frame = -1
    start_time = time.time()
    while current_time < TOTAL_TIME:
        current_time += simulatorController.Forward(NORMAL_SIM_STEP,zoomin_factor_for_collision=1.)
        current_frame = int(current_time/RENDER_STEP)
        if current_frame - previous_frame >= 1:
            position_np = simulator.position.detach().cpu().numpy()
            np.save(folder / "{:04d}.npy".format(current_frame + 1), position_np)
            previous_frame = current_frame
    end_time = time.time()
    result_dict['Normal_bunnyInPipe'] = end_time - start_time




    from bunny_on_pins import simulatorController
    simulatorController.cuda()
    TOTAL_TIME = 600*10*0.001
    NORMAL_SIM_STEP = 0.01
    RENDER_STEP = 0.01

    current_time = 0.0
    previous_frame = -1
    start_time = time.time()
    while current_time < TOTAL_TIME:
        current_time += simulatorController.Forward(NORMAL_SIM_STEP,zoomin_factor_for_collision=2.)
        current_frame = int(current_time/RENDER_STEP)
        if current_frame - previous_frame >= 1:
            position_np = simulator.position.detach().cpu().numpy()
            np.save(folder / "{:04d}.npy".format(current_frame + 1), position_np)
            previous_frame = current_frame
    end_time = time.time()
    result_dict['MP_bunnyOnPins_2'] = end_time - start_time

    from bunny_on_pins import simulatorController
    simulatorController.cuda()
    TOTAL_TIME = 600*10*0.001
    NORMAL_SIM_STEP = 0.01
    RENDER_STEP = 0.005

    current_time = 0.0
    previous_frame = -1
    start_time = time.time()
    while current_time < TOTAL_TIME:
        current_time += simulatorController.Forward(NORMAL_SIM_STEP,zoomin_factor_for_collision=1.)
        current_frame = int(current_time/RENDER_STEP)
        if current_frame - previous_frame >= 1:
            position_np = simulator.position.detach().cpu().numpy()
            np.save(folder / "{:04d}.npy".format(current_frame + 1), position_np)
            previous_frame = current_frame
    end_time = time.time()
    result_dict['Normal_bunnyOnPins'] = end_time - start_time

    return result_dict
if __name__ == "__main__":
    gpu_result = gpu_acc_experiments()
    torch.save(gpu_result,'gpu_acc_result.pt')
    MP_result = motionpro_acc_experiments()
    torch.save(MP_result,'motionpro_acc_result.pt')

# you may use:
# gpu_result = torch.load('gpu_acc_result.pt')
# print(gpu_result) # this would print the dict.