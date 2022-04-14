def prediction(trajectory,num_points=30,avg_points=1):
    #a simple prediction function that predict straight line with constant velocity
    velocity_x_mps = []
    velocity_y_mps = []
    for i in range(1,avg_points+1,1):
        velocity_x_mps.append(trajectory[-i,0] - trajectory[-(1+i),0])
        velocity_y_mps.append(trajectory[-i,1] - trajectory[-(1+i),1])

    velocity_x_mps = np.mean(velocity_x_mps)
    velocity_y_mps = np.mean(velocity_y_mps)

    current_traj = trajectory[-1]
    results = np.zeros((len(trajectory)+num_points,2))

    results[0:len(trajectory)] = trajectory

    for i in range(num_points):
        results[len(trajectory)+i] = np.array([current_traj[0]+velocity_x_mps,current_traj[1]+velocity_y_mps])
        current_traj = results[len(trajectory)+i]
    return results


from argoverse.data_loading.argoverse_forecasting_loader import ArgoverseForecastingLoader

##set root_dir to the correct path to your dataset folder
root_dir = '/research/dept8/jzwang/year1/DenseTNT/val/test/test_obs/data_test/'

afl = ArgoverseForecastingLoader(root_dir)

print('Total number of sequences:',len(afl))

import numpy as np

def get_multiple_forecasts(afl):
    predict_traj = []
    for i in range(9,3,-1):
        predict_traj.append(prediction(afl.agent_traj,avg_points=(i)))
    return np.array(predict_traj)

predict_traj = get_multiple_forecasts(afl)

output_all = {}
counter = 1
for data in afl:
    print('\r'+str(counter)+'/'+str(len(afl)),end="")
    predict_traj = get_multiple_forecasts(data)
    seq_id = int(data.current_seq.name[:-4])
    print("seq_id:", seq_id)
    print("predict_traj[:,20:,:].shape", predict_traj[:,20:,:].shape)
    output_all[seq_id] = predict_traj[:,20:,:]
    counter += 1

from argoverse.evaluation.competition_util import generate_forecasting_h5

output_path = 'competition_files/'
generate_forecasting_h5(output_all, output_path) #this might take awhile
