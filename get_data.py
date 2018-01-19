import numpy as np


def two_qubit_quantum_state_tomography(data):

    ind = 0#(state_index*operations_num*tomography_pulse_num)+(op_index*tomography_pulse_num)

    avgs = []

    avgs.append(1)
    for i in range(15):
        avgs.append(data[ind+i])

    #print "avgs2" + str(avgs2)

    #for testing
    #avgs=data
    #
    #print "avgs" + str(avgs)
    amp = np.sqrt(sum(np.square(avgs)))

    #print amp


    def get_P_array():
        ## Pauli Basis
        I = np.matrix([[1,0],[0,1]])
        X = np.matrix([[0,1],[1,0]])
        Y = np.matrix([[0,-1j],[1j,0]])
        Z = np.matrix([[1,0],[0,-1]])

        P=[]
        P.append(I)
        P.append(X)
        P.append(Y)
        P.append(Z)

        return P

    P = get_P_array()

    # tensor products of Pauli matrixes
    B=[]
    for i in range(4):
        for j in range(4):
            B.append(np.kron(P[i], P[j]))

    den_mat =(0.25*avgs[0]*B[0]).astype(np.complex128)
    for i in np.arange(1,16):
        den_mat += 0.25*avgs[i]*B[i]


    #
    # print("Density Matrix:")
    # print( den_mat)
    # print ("Trace:")
    # print (np.real(np.trace(den_mat)))
    # print ("Density Matrix Squared")
    # print( np.dot(den_mat,den_mat))
    # print( "Trace:")
    # print( np.real(np.trace(np.dot(den_mat,den_mat))))
    #
    #
    #     ### Generate 3D bar chart
    # ## real
    # fig = plt.figure(figsize=(20,20))
    #
    # ax = fig.add_subplot(111, title='Real', projection='3d')
    #
    # coord= [0,1,2,3]
    #
    # x_pos=[]
    # y_pos=[]
    # for i in range(4):
    #     for j in range(4):
    #         x_pos.append(coord[i])
    #         y_pos.append(coord[j])
    #
    # xpos=np.array(x_pos)
    # ypos=np.array(y_pos)
    # zpos=np.array([0]*16)
    # dx = [0.6]*16
    # dy = dx
    # dz=np.squeeze(np.asarray(np.array(np.real(den_mat).flatten())))
    #
    # nrm=mpl.colors.Normalize(-1,1)
    # colors=mpl.cm.Reds(nrm(dz))
    # alphas = np.linspace(0.8, 0.8, len(xpos), endpoint=True)
    #
    # for i in range(len(dx)):
    #     ax.bar3d(xpos[i],ypos[i],zpos[i],dx[i],dy[i],dz[i], alpha=alphas[i],color=colors[i])
    # xticks=['ee','eg','ge','gg']
    # yticks=xticks
    # ax.set_xticks([0,1,2,3])
    # ax.set_xticklabels(xticks)
    # ax.set_yticks([0,1,2,3])
    # ax.set_yticklabels(yticks)
    # ax.set_zlim(-1,1)
    # plt.show()
    #
    # # imaginary
    #
    # fig = plt.figure(figsize=(20,20))
    #
    # ax = fig.add_subplot(111, title='Imaginary', projection='3d')
    #
    # coord= [0,1,2,3]
    #
    # x_pos=[]
    # y_pos=[]
    # for i in range(4):
    #     for j in range(4):
    #         x_pos.append(coord[i])
    #         y_pos.append(coord[j])
    #
    # xpos=np.array(x_pos)
    # ypos=np.array(y_pos)
    # zpos=np.array([0]*16)
    # dx = [0.6]*16
    # dy = dx
    # dz=np.squeeze(np.asarray(np.array(np.imag(den_mat).flatten())))
    #
    #
    # nrm=mpl.colors.Normalize(-1,1)
    # colors=mpl.cm.Reds(nrm(dz))
    # alphas = np.linspace(0.8, 0.8, len(xpos), endpoint=True)
    #
    # for i in range(len(dx)):
    #     ax.bar3d(xpos[i],ypos[i],zpos[i],dx[i],dy[i],dz[i], alpha=alphas[i],color=colors[i])
    # xticks=['ee','eg','ge','gg']
    # yticks=xticks
    # ax.set_xticks([0.3,1.3,2.3,3.3])
    # ax.set_xticklabels(xticks)
    # ax.set_yticks([0.3,1.3,2.3,3.3])
    # ax.set_yticklabels(yticks)
    # ax.set_zlim(-1,1)
    # plt.show()
    #
    # ## absolute value
    #
    # fig = plt.figure(figsize=(20,20))
    #
    # ax = fig.add_subplot(111, title='Abs', projection='3d')
    #
    # coord= [0,1,2,3]
    #
    # x_pos=[]
    # y_pos=[]
    # for i in range(4):
    #     for j in range(4):
    #         x_pos.append(coord[i])
    #         y_pos.append(coord[j])
    #
    # xpos=np.array(x_pos)
    # ypos=np.array(y_pos)
    # zpos=np.array([0]*16)
    # dx = [0.6]*16
    # dy = dx
    # dz=np.squeeze(np.asarray(np.array(np.abs(den_mat).flatten())))
    #
    #
    # nrm=mpl.colors.Normalize(-1,1)
    # colors=mpl.cm.Reds(nrm(dz))
    # alphas = np.linspace(0.8, 0.8, len(xpos), endpoint=True)
    #
    # for i in range(len(dx)):
    #     ax.bar3d(xpos[i],ypos[i],zpos[i],dx[i],dy[i],dz[i], alpha=alphas[i],color=colors[i])
    # xticks=['ee','eg','ge','gg']
    # yticks=xticks
    # ax.set_xticks([0.3,1.3,2.3,3.3])
    # ax.set_xticklabels(xticks)
    # ax.set_yticks([0.3,1.3,2.3,3.3])
    # ax.set_yticklabels(yticks)
    # ax.set_zlim(-1,1)
    # plt.show()

    return den_mat

def data_to_correlators(state_norm):
    IZ = (state_norm[1][0] + state_norm[3][0]) - (state_norm[0][0] + state_norm[2][0])
    ZI = (state_norm[2][0] + state_norm[3][0]) - (state_norm[0][0] + state_norm[1][0])

    IX = (state_norm[1][1] + state_norm[3][1]) - (state_norm[0][1] + state_norm[2][1])
    IY = (state_norm[1][2] + state_norm[3][2]) - (state_norm[0][2] + state_norm[2][2])

    XI = (state_norm[2][3] + state_norm[3][3]) - (state_norm[0][3] + state_norm[1][3])
    YI = (state_norm[2][6] + state_norm[3][6]) - (state_norm[0][6] + state_norm[1][6])

    XX = (state_norm[0][4] + state_norm[3][4]) - (state_norm[1][4] + state_norm[2][4])
    XY = (state_norm[0][5] + state_norm[3][5]) - (state_norm[1][5] + state_norm[2][5])
    YX = (state_norm[0][7] + state_norm[3][7]) - (state_norm[1][7] + state_norm[2][7])
    YY = (state_norm[0][8] + state_norm[3][8]) - (state_norm[1][8] + state_norm[2][8])

    ZZ = (state_norm[0][0] + state_norm[3][0]) - (state_norm[1][0] + state_norm[2][0])


    ZX = (state_norm[0][1] + state_norm[3][1]) - (state_norm[1][1] + state_norm[2][1])
    ZY = (state_norm[0][2] + state_norm[3][2]) - (state_norm[1][2] + state_norm[2][2])

    XZ = (state_norm[0][3] + state_norm[3][3]) - (state_norm[1][3] + state_norm[2][3])
    YZ = (state_norm[0][6] + state_norm[3][6]) - (state_norm[1][6] + state_norm[2][6])


    state_data = [IX,IY,IZ,XI, XX, XY, XZ, YI, YX, YY, YZ, ZI, ZX, ZY, ZZ]

    return state_data


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

        # plt.figure(figsize=(10,7))

        g_cos_ro = data_cos_list[-2]
        g_sin_ro = data_sin_list[-2]
        e_cos_ro = data_cos_list[-1]
        e_sin_ro = data_sin_list[-1]

        # plt.scatter(g_cos_ro,g_sin_ro)
        # plt.scatter(e_cos_ro,e_sin_ro)

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

            # plt.figure(figsize=(10,7))

            g_cos_ro = data_cos_list[-2]
            g_sin_ro = data_sin_list[-2]
            e_cos_ro = data_cos_list[-1]
            e_sin_ro = data_sin_list[-1]

            # plt.scatter(g_cos_ro,g_sin_ro)
            # plt.scatter(e_cos_ro,e_sin_ro)

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

            # plt.figure(figsize=(7,7))
            # plt.title("qubit %s" %qubit_id)
            # plt.plot(data_count_norm[1])

            data_list = data_count_norm[1]

    # print(decision_boundary_list)
    # print(confusion_matrix_list)
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
