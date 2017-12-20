import numpy as np

def get_iq_data(expt_data,expt_pts,het_freq = 0.148, td=0, pi_cal = False):
    data_cos_list=[]
    data_sin_list=[]

    alazar_time_pts = np.arange(len(np.array(expt_data)[0]))

    for data in expt_data:
        cos = np.cos(2*np.pi*het_freq*(alazar_time_pts-td))
        sin = np.sin(2*np.pi*het_freq*(alazar_time_pts-td))

        data_cos = np.dot(data,cos)/len(cos)
        data_sin = np.dot(data,sin)/len(sin)

        data_cos_list.append(data_cos)
        data_sin_list.append(data_sin)

    data_cos_list = np.array(data_cos_list)
    data_sin_list = np.array(data_sin_list)
    data_list = np.sqrt(data_cos_list**2 + data_sin_list**2)

    if pi_cal:
        data_cos_list = (data_cos_list[:-2] - data_cos_list[-2])/(data_cos_list[-1] - data_cos_list[-2])
        data_sin_list = (data_sin_list[:-2] - data_sin_list[-2])/(data_sin_list[-1] - data_sin_list[-2])
        data_list = (data_list[:-2] - data_list[-2])/(data_list[-1] - data_list[-2])


    return data_cos_list, data_sin_list, data_list
