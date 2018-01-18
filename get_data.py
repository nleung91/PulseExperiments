import numpy as np

def get_singleshot_data(expt_data, het_ind ,pi_cal = False):


    data_cos_list= expt_data[het_ind][0]
    data_sin_list= expt_data[het_ind][1]


    if pi_cal:

        ge_cos = np.mean(data_cos_list[-1]) - np.mean(data_cos_list[-2])
        ge_sin = np.mean(data_sin_list[-1]) - np.mean(data_sin_list[-2])

        ge_mean_vec = np.array([ge_cos,ge_sin])

        data_cos_sin_list = np.array([data_cos_list[:-2] - np.mean(data_cos_list[-2]),
                                      data_sin_list[:-2] - np.mean(data_sin_list[-2])])

        data_cos_sin_list = np.transpose(data_cos_sin_list, (1,0,2))


        data_list = np.dot(ge_mean_vec,data_cos_sin_list)/np.dot(ge_mean_vec,ge_mean_vec)


    else:
        cos_contrast = np.abs(np.max(data_cos_list)-np.min(data_cos_list))
        sin_contrast = np.abs(np.max(data_sin_list)-np.min(data_sin_list))

        if cos_contrast > sin_contrast:
            data_list = data_cos_list
        else:
            data_list = data_sin_list


    return data_cos_list, data_sin_list, data_list

def get_singleshot_data_count(expt_data, het_ind ,pi_cal = False):


    data_cos_list= expt_data[het_ind][0]
    data_sin_list= expt_data[het_ind][1]


    if pi_cal:

        ge_cos = np.mean(data_cos_list[-1]) - np.mean(data_cos_list[-2])
        ge_sin = np.mean(data_sin_list[-1]) - np.mean(data_sin_list[-2])

        ge_mean_vec = np.array([ge_cos,ge_sin])

        data_cos_sin_list = np.array([data_cos_list[:-2],
                                      data_sin_list[:-2]])

        data_cos_sin_list = np.transpose(data_cos_sin_list, (1,0,2))


        data_list = np.dot(ge_mean_vec,data_cos_sin_list)/np.dot(ge_mean_vec,ge_mean_vec)

        plt.figure(figsize=(10,7))

        g_cos_ro = data_cos_list[-2]
        g_sin_ro = data_sin_list[-2]
        e_cos_ro = data_cos_list[-1]
        e_sin_ro = data_sin_list[-1]

        plt.scatter(g_cos_ro,g_sin_ro)
        plt.scatter(e_cos_ro,e_sin_ro)

        g_cos_sin_ro = np.array([g_cos_ro,g_sin_ro])
        e_cos_sin_ro = np.array([e_cos_ro,e_sin_ro])

        g_proj = np.dot(ge_mean_vec,g_cos_sin_ro)/np.dot(ge_mean_vec,ge_mean_vec)
        e_proj = np.dot(ge_mean_vec,e_cos_sin_ro)/np.dot(ge_mean_vec,ge_mean_vec)

#         print(np.mean(data_list[0]))
#         print(np.mean(g_proj))

        all_proj = np.array([g_proj,e_proj])
        histo_range = (all_proj.min() / 1.05, all_proj.max() * 1.05)

        g_hist, g_bins = np.histogram(g_proj,bins=1000,range=histo_range)
        e_hist, e_bins = np.histogram(e_proj,bins=1000,range=histo_range)

        g_hist_cumsum = np.cumsum(g_hist)
        e_hist_cumsum = np.cumsum(e_hist)

#         plt.figure(figsize=(7,7))
#         plt.title("qubit %s" %qubit_id)
#         plt.plot(g_bins[:-1],g_hist, 'b')
#         plt.plot(e_bins[:-1],e_hist, 'r')

        max_contrast = abs(((e_hist_cumsum - g_hist_cumsum) / g_hist_cumsum[-1])).max()



        decision_boundary = g_bins[np.argmax(abs(((e_hist_cumsum - g_hist_cumsum) / g_hist_cumsum[-1])))]

#         print(max_contrast)
#         print("decision boundary: %s" %decision_boundary)
#         plt.figure(figsize=(7,7))
#         plt.plot(np.sum(data_list>decision_boundary,axis=1)/data_list.shape[1])
#         plt.figure(figsize=(7,7))
#         plt.title("qubit %s" %qubit_id)
#         plt.plot(np.sum(data_list>decision_boundary,axis=1)/data_list.shape[1])
#         print(data_list.shape)

        confusion_matrix = np.array([[np.sum(g_proj<decision_boundary), np.sum(e_proj<decision_boundary)],
                                     [np.sum(g_proj>decision_boundary),np.sum(e_proj>decision_boundary)]])/data_list.shape[1]


#         print(confusion_matrix)

        confusion_matrix_inv = np.linalg.inv(confusion_matrix)

#         print(confusion_matrix_inv)

        data_count = np.array([np.sum(data_list<decision_boundary,axis=1),
                               np.sum(data_list>decision_boundary,axis=1)])/data_list.shape[1]
#         print(data_count)

        data_count_norm = np.dot(confusion_matrix_inv,data_count)

#         print(data_count_norm)

        # plt.figure(figsize=(7,7))
        # plt.title("qubit %s" %qubit_id)
        # plt.plot(data_count_norm[1])

        data_list = data_count_norm[1]

    else:
        cos_contrast = np.abs(np.max(data_cos_list)-np.min(data_cos_list))
        sin_contrast = np.abs(np.max(data_sin_list)-np.min(data_sin_list))

        if cos_contrast > sin_contrast:
            data_list = data_cos_list
        else:
            data_list = data_sin_list


    return data_cos_list, data_sin_list, data_list

def get_singleshot_data_two_qubits(single_data_list,pi_cal = False):

    decision_boundary_list = []
    confusion_matrix_list = []
    data_list_list = []

    for ii, expt_data in enumerate(single_data_list):

        data_cos_list= expt_data[ii][0]
        data_sin_list= expt_data[ii][1]


        if pi_cal:

            ge_cos = np.mean(data_cos_list[-1]) - np.mean(data_cos_list[-2])
            ge_sin = np.mean(data_sin_list[-1]) - np.mean(data_sin_list[-2])

            ge_mean_vec = np.array([ge_cos,ge_sin])

            data_cos_sin_list = np.array([data_cos_list[:-2],
                                          data_sin_list[:-2]])

            data_cos_sin_list = np.transpose(data_cos_sin_list, (1,0,2))


            data_list = np.dot(ge_mean_vec,data_cos_sin_list)/np.dot(ge_mean_vec,ge_mean_vec)

            data_list_list.append(data_list)

            plt.figure(figsize=(10,7))

            g_cos_ro = data_cos_list[-2]
            g_sin_ro = data_sin_list[-2]
            e_cos_ro = data_cos_list[-1]
            e_sin_ro = data_sin_list[-1]

            plt.scatter(g_cos_ro,g_sin_ro)
            plt.scatter(e_cos_ro,e_sin_ro)

            g_cos_sin_ro = np.array([g_cos_ro,g_sin_ro])
            e_cos_sin_ro = np.array([e_cos_ro,e_sin_ro])

            g_proj = np.dot(ge_mean_vec,g_cos_sin_ro)/np.dot(ge_mean_vec,ge_mean_vec)
            e_proj = np.dot(ge_mean_vec,e_cos_sin_ro)/np.dot(ge_mean_vec,ge_mean_vec)

    #         print(np.mean(data_list[0]))
    #         print(np.mean(g_proj))

            all_proj = np.array([g_proj,e_proj])
            histo_range = (all_proj.min() / 1.05, all_proj.max() * 1.05)

            g_hist, g_bins = np.histogram(g_proj,bins=1000,range=histo_range)
            e_hist, e_bins = np.histogram(e_proj,bins=1000,range=histo_range)

            g_hist_cumsum = np.cumsum(g_hist)
            e_hist_cumsum = np.cumsum(e_hist)

    #         plt.figure(figsize=(7,7))
    #         plt.title("qubit %s" %qubit_id)
    #         plt.plot(g_bins[:-1],g_hist, 'b')
    #         plt.plot(e_bins[:-1],e_hist, 'r')

            max_contrast = abs(((e_hist_cumsum - g_hist_cumsum) / g_hist_cumsum[-1])).max()



            decision_boundary = g_bins[np.argmax(abs(((e_hist_cumsum - g_hist_cumsum) / g_hist_cumsum[-1])))]

            decision_boundary_list.append(decision_boundary)

    #         print(max_contrast)
    #         print("decision boundary: %s" %decision_boundary)
    #         plt.figure(figsize=(7,7))
    #         plt.plot(np.sum(data_list>decision_boundary,axis=1)/data_list.shape[1])
    #         plt.figure(figsize=(7,7))
    #         plt.title("qubit %s" %qubit_id)
    #         plt.plot(np.sum(data_list>decision_boundary,axis=1)/data_list.shape[1])
    #         print(data_list.shape)

            confusion_matrix = np.array([[np.sum(g_proj<decision_boundary), np.sum(e_proj<decision_boundary)],
                                         [np.sum(g_proj>decision_boundary),np.sum(e_proj>decision_boundary)]])/data_list.shape[1]

            confusion_matrix_list.append(confusion_matrix)
    #         print(confusion_matrix)

            confusion_matrix_inv = np.linalg.inv(confusion_matrix)

    #         print(confusion_matrix_inv)

            data_count = np.array([np.sum(data_list<decision_boundary,axis=1),
                                   np.sum(data_list>decision_boundary,axis=1)])/data_list.shape[1]
    #         print(data_count)

            data_count_norm = np.dot(confusion_matrix_inv,data_count)

    #         print(data_count_norm)

            plt.figure(figsize=(7,7))
            plt.title("qubit %s" %qubit_id)
            plt.plot(data_count_norm[1])

            data_list = data_count_norm[1]

    print(decision_boundary_list)
    print(confusion_matrix_list)
    gg = np.sum(np.bitwise_and((data_list_list[0] < decision_boundary_list[0]) ,
                               (data_list_list[1] < decision_boundary_list[1])),axis=1)/data_list_list[0].shape[1]
    ge = np.sum(np.bitwise_and((data_list_list[0] < decision_boundary_list[0]) ,
                               (data_list_list[1] > decision_boundary_list[1])),axis=1)/data_list_list[0].shape[1]
    eg = np.sum(np.bitwise_and((data_list_list[0] > decision_boundary_list[0]) ,
                               (data_list_list[1] < decision_boundary_list[1])),axis=1)/data_list_list[0].shape[1]
    ee = np.sum(np.bitwise_and((data_list_list[0] > decision_boundary_list[0]) ,
                               (data_list_list[1] > decision_boundary_list[1])),axis=1)/data_list_list[0].shape[1]

    total_confusion_matrix = np.kron(confusion_matrix_list[0],confusion_matrix_list[1])

    total_confusion_matrix_inv = np.linalg.inv(total_confusion_matrix)
    state = np.array([gg,ge,eg,ee])

    state_norm = np.dot(total_confusion_matrix_inv,state)

    return state_norm

def get_iq_data(expt_data,het_freq = 0.148, td=0, pi_cal = False):
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

    if pi_cal:

        ge_cos = data_cos_list[-1] - data_cos_list[-2]
        ge_sin = data_sin_list[-1] - data_sin_list[-2]

        ge_mean_vec = np.array([ge_cos,ge_sin])

        data_cos_sin_list = np.array([data_cos_list[:-2] - data_cos_list[-2],data_sin_list[:-2] - data_sin_list[-2]])

        data_list = np.dot(ge_mean_vec,data_cos_sin_list)/np.dot(ge_mean_vec,ge_mean_vec)


    else:
        cos_contrast = np.abs(np.max(data_cos_list)-np.min(data_cos_list))
        sin_contrast = np.abs(np.max(data_sin_list)-np.min(data_sin_list))

        if cos_contrast > sin_contrast:
            data_list = data_cos_list
        else:
            data_list = data_sin_list

    return data_cos_list, data_sin_list, data_list
