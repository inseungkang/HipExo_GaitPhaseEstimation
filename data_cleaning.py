from data_processing import *

sensors = ['lJPos', 'rJPos', 'lJVel',
           'rJVel', 'gyroX', 'gyroY', 'gyroZ', 'accX',
           'accY', 'accZ', 'nWalk']

# def manual_label_data(subject):
def manual_label_data(data, filename):
    """For each trial, plot the joint position and detected the peaks, when
    click on the graph, the x-coordinate of the mouse click will be printed and the
    plot will be closed. Then user need to enter "add {int: x-coord}" to add a
    peak, "rm {int: x-coord}" to remove the nearest peak, or press enter to move on
    to the next trial.
    A plot with both left and right joint will be displayed, when that plot is
    closed, a file containing the labeled data will be saved to the
    correspond subject's data folder.

    Args:
        subject (int): subject number
    """
    def onclick(event):
        print(event.xdata)
        # plt.close()

    # file_path = f'data/AB{subject:02d}/' + '*' + "ZI*.npy"
    # file_path = 'data/evalData/AB1_ML.txt'
    # Read data
    # for file in sorted(glob.glob(file_path)):
    #     data = np.load(file)
    #     data = pd.DataFrame(data, columns=headers)

        # # drop the 32nd column which only contains NaN values
        # data.dropna(axis=1, inplace=True)

        # only keep the 10 sensors data columns + nWalk, lGC, rGC
        # data = data[sensors]

    lJPos, rJPos = extract_joint_positions([data])
    lMaximas, rMaximas = find_local_maximas(lJPos[0]), find_local_maximas(rJPos[0])

    # Plot graph, onclick -> print x coordinate and close graph
    # Get input from uesr to add peak, delete peak, or move on to the next plot
    f = plt.figure(figsize=(10, 4))
    # plt.title(file + ' Left')
    plt.plot(lJPos[0])
    # plt.plot(data['m_lCtrlTrq'])
    # plt.plot(data['leftWalkMode'])
    plt.plot(lMaximas, [lJPos[0][i] for i in lMaximas], 'r*')
    f.canvas.mpl_connect('button_press_event', onclick)
    plt.show()
    val = input("Press Enter if ok; \nType 'rm {int}' to remove a maxima; \nType 'add {int}' to add a maxima:\n")
    while val:
        try:
            plt.close()
            val = val.split(' ')
            if val[0] == 'rm':
                closest = min(lMaximas, key=lambda x : abs(x-int(val[1])))
                lMaximas.remove(closest)
            elif val[0] == 'add':
                lMaximas.append(int(val[1]))
                lMaximas.sort()
            else:
                print("Invalid Input!!!")
                raise Exception
            f = plt.figure(figsize=(10, 4))
            # plt.title(file + ' Left')
            plt.plot(lJPos[0])
            # plt.plot(data['m_lCtrlTrq'])
            # plt.plot(data['leftWalkMode'])
            plt.plot(lMaximas, [lJPos[0][i] for i in lMaximas], 'r*')
            f.canvas.mpl_connect('button_press_event', onclick)
            plt.show()
            val = input("Press Enter if ok; \nType 'rm {int}' to remove a maxima; \nType 'add {int}' to add a maxima:\n")
            continue
        except Exception:
            print("Something went wrong >.<")
            print(sys.exc_info())
            val = input("Press Enter if ok; \nType 'rm {int}' to remove a maxima; \nType 'add {int}' to add a maxima:\n")
            continue

    f = plt.figure(figsize=(10, 4))
    # plt.title(file + ' Right')
    plt.plot(rJPos[0])
    plt.plot(rMaximas, [rJPos[0][i] for i in rMaximas], 'r*')
    f.canvas.mpl_connect('button_press_event', onclick)
    plt.show()

    val = input("Press Enter if ok; \nType 'rm {int}' to remove a maxima; \nType 'add {int}' to add a maxima:\n")
    while val:
        try:
            plt.close()
            val = val.split(' ')
            if val[0] == 'rm':
                closest = min(rMaximas, key=lambda x : abs(x-int(val[1])))
                rMaximas.remove(closest)
                print(f"Removed point {closest}")
            elif val[0] == 'add':
                rMaximas.append(int(val[1]))
                rMaximas.sort()
                print(f"Added point " + val[1])
            else:
                print("Invalid Input!!!")
                continue
            f = plt.figure(figsize=(10, 4))
            # plt.title(file + ' Right')
            plt.plot(rJPos[0])
            plt.plot(rMaximas, [rJPos[0][i] for i in rMaximas], 'r*')
            f.canvas.mpl_connect('button_press_event', onclick)
            plt.show()
            val = input("Press Enter if ok; \nType 'rm {int}' to remove a maxima; \nType 'add {int}' to add a maxima:\n")
            continue
        except Exception:
            print("Something went wrong >.<")
            print(sys.exc_info())
            val = input("Press Enter if ok; \nType 'rm {int}' to remove a maxima; \nType 'add {int}' to add a maxima:\n")
            continue
    # Mark label as 1 at maximas and 0 at maxima+1
    lY = pd.Series(np.nan, index=range(0, data.shape[0]))
    rY = pd.Series(np.nan, index=range(0, data.shape[0]))
    for maxima in lMaximas:
        lY[maxima] = 1
        lY[maxima+1] = 0
    for maxima in rMaximas:
        rY[maxima] = 1
        rY[maxima+1] = 0

    # Linearly interpolate the labels between every 0 and 1 and fill in the
    # rest with 0's
    # Conver to polar coordinates
    lY.interpolate(inplace=True), rY.interpolate(inplace=True)
    lY.fillna(0, inplace=True), rY.fillna(0, inplace=True)
    ly_theta, ry_theta = lY * 2 * np.pi, rY * 2 * np.pi
    left_x, left_y = np.cos(ly_theta), np.sin(ly_theta)
    right_x, right_y = np.cos(ry_theta), np.sin(ry_theta)
    labels = pd.DataFrame({'leftGaitPhaseX': left_x, 'leftGaitPhaseY': left_y,
                                'rightGaitPhaseX': right_x, 'rightGaitPhaseY': right_y})

    # Combine the data and the labels
    data[labels.columns] = labels
    all_maximas = sorted(lMaximas + rMaximas)
    all_maximas = all_maximas[1:-1]

    if lMaximas[0]<rMaximas[0]:
        lMaximas = lMaximas[1:]
    else:
        rMaximas = rMaximas[1:]

    if lMaximas[-1]>rMaximas[-1]:
        lMaximas = lMaximas[:-1]
    else:
        rMaximas = rMaximas[:-1]

    data = data.iloc[all_maximas[0]:all_maximas[-1]+1, :]

    # Plot both left and right joint as well as the final peaks
    # f = plt.figure(figsize=(10, 7))
    # plt.subplot(211)
    # # plt.title(file + ' Left')
    # plt.title(' Left')
    # plt.plot(data['lJPos'])
    # # plt.vlines([i for i, v in enumerate(data['lGC']) if v == 0], -1, 1, 'r')
    # plt.plot(lMaximas, [data['lJPos'][i] for i in lMaximas], 'r*')

    # plt.subplot(212)
    # # plt.title(file + ' Right')
    # plt.title(' Right')
    # plt.plot(data['rJPos'])
    # # plt.vlines([i for i, v in enumerate(data['rGC']) if v == 0], -1, 1, 'r')
    # plt.plot(rMaximas, [data['rJPos'][i] for i in rMaximas], 'r*')

    # f.canvas.mpl_connect('button_press_event', onclick)
    # plt.show()

    # Save the labeled data file to the corresponding folder
    # Each file should be shape (M, 15) -> 10 sensors + nWalk + 4 labels
    # np.savetxt(file[:10] + 'labeled_' + file[10:-4], data)
    # np.savetxt('data/evalData/labeled.txt', data)
    data.to_csv(filename)
    # print("You just finihsed 1 trial! Yay!")
    # print("Labeled file saved as " + file[:10] + 'labeled_' + file[10:-4] + "\n\n")
    return data

def manual_scrap_data(data, filename):
    """For each trial, plot the joint position and detected the peaks, when
    click on the graph, the x-coordinate of the mouse click will be printed and the
    plot will be closed. Then user need to enter "add {int: x-coord}" to add a
    peak, "rm {int: x-coord}" to remove the nearest peak, or press enter to move on
    to the next trial.
    A plot with both left and right joint will be displayed, when that plot is
    closed, a file containing the labeled data will be saved to the
    correspond subject's data folder.

    Args:
        subject (int): subject number
    """
    def convert_to_gp(x, y):
        gp = np.mod(np.arctan2(y, x)+2*np.pi, 2*np.pi) / (2*np.pi)
        return gp

    def onclick(event):
        print(event.xdata)

    def plot_trial(l_data, start=False, end=False):
        # f, axs = plt.subplots(2,1, sharex=True, figsize=(10, 4))
        # f.suptitle(file)
        # axs[0].plot(lJPos[0])
        # axs[1].plot(rJPos[0])
        # if start != False:
        #     axs[0].axvline(start, color='r')
        #     axs[1].axvline(start, color='r')
        # if end != False:
        #     axs[0].axvline(end, color='r')
        #     axs[1].axvline(end, color='r')
        # axs[0].set_title('Left Hip Angle')
        # axs[1].set_title('Right Hip Angle')
        # f.canvas.mpl_connect('button_press_event', onclick)
        # plt.show()
        f = plt.figure(figsize=(10, 4))
        l_gp = convert_to_gp(l_data['leftGaitPhaseX'], l_data['leftGaitPhaseY'])
        plt.plot(l_gp, label='ground_truth')
        # plt.plot(l_data['leftJointPosition'], label='Joint Position')
        plt.plot(l_data['leftGaitPhase'], alpha=0.7, label='predicted')
        plt.plot(l_data['leftWalkMode'], label='mode')
        plt.legend(loc=9)
        if start != False:
            plt.axvline(start, color='r')
        if end != False:
            plt.axvline(end, color='r')
        f.canvas.mpl_connect('button_press_event', onclick)
        plt.show()    

    # file_path = f'data/AB{subject:02d}/' + '*' + "BT*.npy"
    # file = glob.glob(file_path)[0]
    # # Read data
    # for file in sorted(glob.glob(file_path)):
    #     data = np.load(file)
    #     data = pd.DataFrame(data, columns=columns)
    #     # drop the 32nd column which only contains NaN values
    #     data.dropna(axis=1, inplace=True)

    #     # only keep the 10 sensors data columns + nWalk, lGC, rGC
    #     data = data[sensors]

    # lJPos, rJPos = extract_joint_positions([data])
    # lMaximas, rMaximas = find_local_maximas(lJPos[0]), find_local_maximas(rJPos[0])

    # Data for this file
    current_file_subsets = []

    # Plot graph, onclick -> print x coordinate and close graph
    # Get input from uesr to add peak, delete peak, or move on to the next plot

    print('Click anywhere to continue')
    plot_trial(data)
    val = input("Press 1 to keep all data\nPress 2 to discard all data\nPress 3 to input a starting point\n")
    if (val == "1"):
        # Save all data for this trial
        np.savetxt(filename, data)
    elif (val == "2"):
        print('Done with a trial!')
    else:
        # Repeat the following until 1 or 2 is pressed
        # - Show the current remaining data
        # - Ask for a starting point
        # - Ask for an ending point or press X to restart
        # - If restart, continue loop
        # - If ending point clicked, ask to confirm range of data by pressing enter or restart by pressing X
        # - If X, continue loop
        # - If enter, add start and end indices to current_file_subsets

        remaining_data = data
        while (val != "1" and val != "2"):
            print('Click to select a starting point')
            # Input a starting point
            plot_trial(remaining_data)
            val = input('Enter the starting point\n')
            start_ind = int(val)
            print('Click to select an ending point')
            plot_trial(remaining_data, start_ind)

            val = input('Enter the ending point or press x to restart\n')
            if (val == "x"):
                print('Restart trial')
                continue
            else:
                end_ind = int(val)
                print('Review the range. Click to respond')
                plot_trial(remaining_data, start_ind, end_ind)
                val = input('Confirm range by pressing enter or restart by pressing x\n')
                if (val == "x"):
                    print('Restart trial')
                    continue
                else:
                    current_file_subsets.append((start_ind, end_ind))
                    remaining_data = remaining_data.loc[end_ind:]
                    print('Click anywhere to continue')
                    plot_trial(remaining_data)
                    val = input("Press 1 to keep all data\nPress 2 to discard all data\nPress 3 to input a starting point\n")

        for i, (start, end) in enumerate(current_file_subsets, start=1):
            clip = data.loc[start:end]

            # Save all data for this trial
            np.savetxt(filename+f'_{i}', clip)
            print('Saved')

    print('Done with a trial!')

def manual_label_chopped_data(subject):
    """For each trial, plot the joint position and detected the peaks, when
    click on the graph, the x-coordinate of the mouse click will be printed and the
    plot will be closed. Then user need to enter "add {int: x-coord}" to add a
    peak, "rm {int: x-coord}" to remove the nearest peak, or press enter to move on
    to the next trial.
    A plot with both left and right joint will be displayed, when that plot is
    closed, a file containing the labeled data will be saved to the
    correspond subject's data folder.

    Args:
        subject (int): subject number
    """
    def onclick(event):
        print(event.xdata)
        plt.close()

    file_path = f'data/AB{subject:02d}/chopped' + "*" + "BT*"
    # Read data
    for file in sorted(glob.glob(file_path)):
        data = np.loadtxt(file)
        data = pd.DataFrame(data, columns=sensors)

        lJPos, rJPos = extract_joint_positions([data])
        lMaximas, rMaximas = find_local_maximas(lJPos[0]), find_local_maximas(rJPos[0])

        # Plot graph, onclick -> print x coordinate and close graph
        # Get input from uesr to add peak, delete peak, or move on to the next plot
        f = plt.figure(figsize=(10, 4))
        plt.title(file + ' Left')
        plt.plot(lJPos[0])
        plt.plot(lMaximas, [lJPos[0][i] for i in lMaximas], 'r*')
        f.canvas.mpl_connect('button_press_event', onclick)
        plt.show()
        val = input("Press Enter if ok; \nType 'rm {int}' to remove a maxima; \nType 'add {int}' to add a maxima:\n")
        while val:
            try:
                plt.close()
                val = val.split(' ')
                if val[0] == 'rm':
                    closest = min(lMaximas, key=lambda x : abs(x-int(val[1])))
                    lMaximas.remove(closest)
                elif val[0] == 'add':
                    lMaximas.append(int(val[1]))
                    lMaximas.sort()
                else:
                    print("Invalid Input!!!")
                    raise Exception
                f = plt.figure(figsize=(10, 4))
                plt.title(file + ' Left')
                plt.plot(lJPos[0])
                plt.plot(lMaximas, [lJPos[0][i] for i in lMaximas], 'r*')
                f.canvas.mpl_connect('button_press_event', onclick)
                plt.show()
                val = input("Press Enter if ok; \nType 'rm {int}' to remove a maxima; \nType 'add {int}' to add a maxima:\n")
                continue
            except Exception:
                print("Something went wrong >.<")
                print(sys.exc_info())
                val = input("Press Enter if ok; \nType 'rm {int}' to remove a maxima; \nType 'add {int}' to add a maxima:\n")
                continue

        f = plt.figure(figsize=(10, 4))
        plt.title(file + ' Right')
        plt.plot(rJPos[0])
        plt.plot(rMaximas, [rJPos[0][i] for i in rMaximas], 'r*')
        f.canvas.mpl_connect('button_press_event', onclick)
        plt.show()

        val = input("Press Enter if ok; \nType 'rm {int}' to remove a maxima; \nType 'add {int}' to add a maxima:\n")
        while val:
            try:
                plt.close()
                val = val.split(' ')
                if val[0] == 'rm':
                    closest = min(rMaximas, key=lambda x : abs(x-int(val[1])))
                    rMaximas.remove(closest)
                    print(f"Removed point {closest}")
                elif val[0] == 'add':
                    rMaximas.append(int(val[1]))
                    rMaximas.sort()
                    print(f"Added point " + val[1])
                else:
                    print("Invalid Input!!!")
                    continue
                f = plt.figure(figsize=(10, 4))
                plt.title(file + ' Right')
                plt.plot(rJPos[0])
                plt.plot(rMaximas, [rJPos[0][i] for i in rMaximas], 'r*')
                f.canvas.mpl_connect('button_press_event', onclick)
                plt.show()
                val = input("Press Enter if ok; \nType 'rm {int}' to remove a maxima; \nType 'add {int}' to add a maxima:\n")
                continue
            except Exception:
                print("Something went wrong >.<")
                print(sys.exc_info())
                val = input("Press Enter if ok; \nType 'rm {int}' to remove a maxima; \nType 'add {int}' to add a maxima:\n")
                continue
        # Mark label as 1 at maximas and 0 at maxima+1
        lY = pd.Series(np.nan, index=range(0, data.shape[0]))
        rY = pd.Series(np.nan, index=range(0, data.shape[0]))
        for maxima in lMaximas:
            lY[maxima] = 1
            lY[maxima+1] = 0
        for maxima in rMaximas:
            rY[maxima] = 1
            rY[maxima+1] = 0

        # Linearly interpolate the labels between every 0 and 1 and fill in the
        # rest with 0's
        # Conver to polar coordinates
        lY.interpolate(inplace=True), rY.interpolate(inplace=True)
        lY.fillna(0, inplace=True), rY.fillna(0, inplace=True)
        ly_theta, ry_theta = lY * 2 * np.pi, rY * 2 * np.pi
        left_x, left_y = np.cos(ly_theta), np.sin(ly_theta)
        right_x, right_y = np.cos(ry_theta), np.sin(ry_theta)
        labels = pd.DataFrame({'leftGaitPhaseX': left_x, 'leftGaitPhaseY': left_y,
                                 'rightGaitPhaseX': right_x, 'rightGaitPhaseY': right_y})

        # Combine the data and the labels
        data[labels.columns] = labels
        all_maximas = sorted(lMaximas + rMaximas)
        all_maximas = all_maximas[1:-1]

        if lMaximas[0]<rMaximas[0]:
            lMaximas = lMaximas[1:]
        else:
            rMaximas = rMaximas[1:]

        if lMaximas[-1]>rMaximas[-1]:
            lMaximas = lMaximas[:-1]
        else:
            rMaximas = rMaximas[:-1]

        data = data.iloc[all_maximas[0]:all_maximas[-1]+1, :]

        # Plot both left and right joint as well as the final peaks
        f = plt.figure(figsize=(10, 7))
        plt.subplot(211)
        plt.title(file + ' Left')
        plt.plot(data['lJPos'])
        # plt.vlines([i for i, v in enumerate(data['lGC']) if v == 0], -1, 1, 'r')
        plt.plot(lMaximas, [data['lJPos'][i] for i in lMaximas], 'r*')

        plt.subplot(212)
        plt.title(file + ' Right')
        plt.plot(data['rJPos'])
        # plt.vlines([i for i, v in enumerate(data['rGC']) if v == 0], -1, 1, 'r')
        plt.plot(rMaximas, [data['rJPos'][i] for i in rMaximas], 'r*')

        f.canvas.mpl_connect('button_press_event', onclick)
        plt.show()

        # Save the labeled data file to the corresponding folder
        # Each file should be shape (M, 15) -> 10 sensors + nWalk + 4 labels
        np.savetxt(file[:10] + 'labeled_' + file[10:], data)
        print("You just finihsed 1 trial! Yay!")
        print("Labeled file saved as " + file[:10] + 'labeled_' + file[10:] + "\n\n")

def manual_label_data_local_min(data):
    """For each trial, plot the joint position and detected the peaks, when
    click on the graph, the x-coordinate of the mouse click will be printed and the
    plot will be closed. Then user need to enter "add {int: x-coord}" to add a
    peak, "rm {int: x-coord}" to remove the nearest peak, or press enter to move on
    to the next trial.
    A plot with both left and right joint will be displayed, when that plot is
    closed, a file containing the labeled data will be saved to the
    correspond subject's data folder.

    Args:
        subject (int): subject number
    """
    def onclick(event):
        print(event.xdata)
        # plt.close()

    # file_path = f'data/AB{subject:02d}/' + '*' + "ZI*.npy"
    # file_path = 'data/evalData/AB1_ML.txt'
    # Read data
    # for file in sorted(glob.glob(file_path)):
    #     data = np.load(file)
    #     data = pd.DataFrame(data, columns=headers)

        # # drop the 32nd column which only contains NaN values
        # data.dropna(axis=1, inplace=True)

        # only keep the 10 sensors data columns + nWalk, lGC, rGC
        # data = data[sensors]

    lJPos, rJPos = extract_joint_positions([data])
    lMinimas, rMinimas = find_local_maximas(-lJPos[0]), find_local_maximas(-rJPos[0])

    # Plot graph, onclick -> print x coordinate and close graph
    # Get input from uesr to add peak, delete peak, or move on to the next plot
    f = plt.figure(figsize=(10, 4))
    # plt.title(file + ' Left')
    plt.plot(lJPos[0])
    plt.plot(lMinimas, [lJPos[0][i] for i in lMinimas], 'r*')
    f.canvas.mpl_connect('button_press_event', onclick)
    plt.show()
    val = input("Press Enter if ok; \nType 'rm {int}' to remove a minima; \nType 'add {int}' to add a minima:\n")
    while val:
        try:
            plt.close()
            val = val.split(' ')
            if val[0] == 'rm':
                closest = min(lMinimas, key=lambda x : abs(x-int(val[1])))
                lMinimas.remove(closest)
            elif val[0] == 'add':
                lMinimas.append(int(val[1]))
                lMinimas.sort()
            else:
                print("Invalid Input!!!")
                raise Exception
            f = plt.figure(figsize=(10, 4))
            # plt.title(file + ' Left')
            plt.plot(lJPos[0])
            plt.plot(lMinimas, [lJPos[0][i] for i in lMinimas], 'r*')
            f.canvas.mpl_connect('button_press_event', onclick)
            plt.show()
            val = input("Press Enter if ok; \nType 'rm {int}' to remove a minima; \nType 'add {int}' to add a minima:\n")
            continue
        except Exception:
            print("Something went wrong >.<")
            print(sys.exc_info())
            val = input("Press Enter if ok; \nType 'rm {int}' to remove a minima; \nType 'add {int}' to add a minima:\n")
            continue

    f = plt.figure(figsize=(10, 4))
    # plt.title(file + ' Right')
    plt.plot(rJPos[0])
    plt.plot(rMinimas, [rJPos[0][i] for i in rMinimas], 'r*')
    f.canvas.mpl_connect('button_press_event', onclick)
    plt.show()

    val = input("Press Enter if ok; \nType 'rm {int}' to remove a minima; \nType 'add {int}' to add a minima:\n")
    while val:
        try:
            plt.close()
            val = val.split(' ')
            if val[0] == 'rm':
                closest = min(rMinimas, key=lambda x : abs(x-int(val[1])))
                rMinimas.remove(closest)
                print(f"Removed point {closest}")
            elif val[0] == 'add':
                rMinimas.append(int(val[1]))
                rMinimas.sort()
                print(f"Added point " + val[1])
            else:
                print("Invalid Input!!!")
                continue
            f = plt.figure(figsize=(10, 4))
            # plt.title(file + ' Right')
            plt.plot(rJPos[0])
            plt.plot(rMinimas, [rJPos[0][i] for i in rMinimas], 'r*')
            f.canvas.mpl_connect('button_press_event', onclick)
            plt.show()
            val = input("Press Enter if ok; \nType 'rm {int}' to remove a minima; \nType 'add {int}' to add a minima:\n")
            continue
        except Exception:
            print("Something went wrong >.<")
            print(sys.exc_info())
            val = input("Press Enter if ok; \nType 'rm {int}' to remove a minima; \nType 'add {int}' to add a minima:\n")
            continue
    # Mark label as 1 at minimas and 0 at minima+1
    lY = pd.Series(np.nan, index=range(0, data.shape[0]))
    rY = pd.Series(np.nan, index=range(0, data.shape[0]))
    l_diff = np.diff(lMinimas)
    r_diff = np.diff(rMinimas)
    for i in range(len(lMinimas)-1):
        lY[lMinimas[i]] = 0.4
        lY[int(lMinimas[i]+ l_diff[i]*0.6)] = 1
        lY[int(lMinimas[i] + l_diff[i]*0.6 + 1)] = 0
    for i in range(len(rMinimas)-1):
        rY[rMinimas[i]] = 0.4
        rY[int(rMinimas[i]+ r_diff[i]*0.6)] = 1
        rY[int(rMinimas[i] + r_diff[i]*0.6 + 1)] = 0
    # Linearly interpolate the labels between every 0 and 1 and fill in the
    # rest with 0's
    # Conver to polar coordinates
    lY.interpolate(inplace=True), rY.interpolate(inplace=True)
    lY.fillna(0, inplace=True), rY.fillna(0, inplace=True)
    ly_theta, ry_theta = lY * 2 * np.pi, rY * 2 * np.pi
    left_x, left_y = np.cos(ly_theta), np.sin(ly_theta)
    right_x, right_y = np.cos(ry_theta), np.sin(ry_theta)
    labels = pd.DataFrame({'leftGaitPhaseX': left_x, 'leftGaitPhaseY': left_y,
                                'rightGaitPhaseX': right_x, 'rightGaitPhaseY': right_y})

    # Combine the data and the labels
    data[labels.columns] = labels

    all_minimas = sorted(lMinimas + rMinimas)
    all_minimas = all_minimas[1:-1]

    if lMinimas[0]<rMinimas[0]:
        lMinimas = lMinimas[1:]
    else:
        rMinimas = rMinimas[1:]

    if lMinimas[-1]>rMinimas[-1]:
        lMinimas = lMinimas[:-1]
    else:
        rMinimas = rMinimas[:-1]

    data = data.iloc[all_minimas[0]:all_minimas[-1]+1, :]

    # Plot both left and right joint as well as the final peaks
    # f = plt.figure(figsize=(10, 7))
    # plt.subplot(211)
    # # plt.title(file + ' Left')
    # plt.title(' Left')
    # plt.plot(data['lJPos'])
    # # plt.vlines([i for i, v in enumerate(data['lGC']) if v == 0], -1, 1, 'r')
    # plt.plot(lMinimas, [data['lJPos'][i] for i in lMinimas], 'r*')

    # plt.subplot(212)
    # # plt.title(file + ' Right')
    # plt.title(' Right')
    # plt.plot(data['rJPos'])
    # # plt.vlines([i for i, v in enumerate(data['rGC']) if v == 0], -1, 1, 'r')
    # plt.plot(rMinimas, [data['rJPos'][i] for i in rMinimas], 'r*')

    # f.canvas.mpl_connect('button_press_event', onclick)
    # plt.show()

    # Save the labeled data file to the corresponding folder
    # Each file should be shape (M, 15) -> 10 sensors + nWalk + 4 labels
    # np.savetxt(file[:10] + 'labeled_' + file[10:-4], data)
    # np.savetxt('data/evalData/labeled.txt', data)
    data.to_csv('data/evalData/labeled.txt')
    # print("You just finihsed 1 trial! Yay!")
    # print("Labeled file saved as " + file[:10] + 'labeled_' + file[10:-4] + "\n\n")
    return data

def manual_label_data_eval(data, filename):
    def onclick(event):
        print(event.xdata)
        # plt.close()

    lJPos, rJPos = extract_joint_positions([data])
    lMaximas, rMaximas = find_local_maximas(lJPos[0]), find_local_maximas(rJPos[0])
    lMinimas, rMinimas = find_local_maximas(-lJPos[0]), find_local_maximas(-rJPos[0])

    f = plt.figure(figsize=(10, 4))
    # plt.title(file + ' Left')
    plt.plot(lJPos[0])
    plt.plot(data['leftWalkMode'])
    plt.plot(lMaximas, [lJPos[0][i] for i in lMaximas], 'r*')
    f.canvas.mpl_connect('button_press_event', onclick)
    plt.show()
    val = input("Press Enter if ok; \nType 'rm {int}' to remove a maxima; \nType 'add {int}' to add a maxima:\n")
    while val:
        try:
            plt.close()
            val = val.split(' ')
            if val[0] == 'rm':
                closest = min(lMaximas, key=lambda x : abs(x-int(val[1])))
                lMaximas.remove(closest)
            elif val[0] == 'add':
                lMaximas.append(int(val[1]))
                lMaximas.sort()
            else:
                print("Invalid Input!!!")
                raise Exception
            f = plt.figure(figsize=(10, 4))
            # plt.title(file + ' Left')
            plt.plot(lJPos[0])
            plt.plot(data['leftWalkMode'])
            plt.plot(lMaximas, [lJPos[0][i] for i in lMaximas], 'r*')
            f.canvas.mpl_connect('button_press_event', onclick)
            plt.show()
            val = input("Press Enter if ok; \nType 'rm {int}' to remove a maxima; \nType 'add {int}' to add a maxima:\n")
            continue
        except Exception:
            print("Something went wrong >.<")
            print(sys.exc_info())
            val = input("Press Enter if ok; \nType 'rm {int}' to remove a maxima; \nType 'add {int}' to add a maxima:\n")
            continue

    f = plt.figure(figsize=(10, 4))
    # plt.title(file + ' Right')
    plt.plot(rJPos[0])
    plt.plot(data['rightWalkMode'])
    plt.plot(rMaximas, [rJPos[0][i] for i in rMaximas], 'r*')
    f.canvas.mpl_connect('button_press_event', onclick)
    plt.show()

    val = input("Press Enter if ok; \nType 'rm {int}' to remove a maxima; \nType 'add {int}' to add a maxima:\n")
    while val:
        try:
            plt.close()
            val = val.split(' ')
            if val[0] == 'rm':
                closest = min(rMaximas, key=lambda x : abs(x-int(val[1])))
                rMaximas.remove(closest)
                print(f"Removed point {closest}")
            elif val[0] == 'add':
                rMaximas.append(int(val[1]))
                rMaximas.sort()
                print(f"Added point " + val[1])
            else:
                print("Invalid Input!!!")
                continue
            f = plt.figure(figsize=(10, 4))
            # plt.title(file + ' Right')
            plt.plot(rJPos[0])
            plt.plot(data['rightWalkMode'])
            plt.plot(rMaximas, [rJPos[0][i] for i in rMaximas], 'r*')
            f.canvas.mpl_connect('button_press_event', onclick)
            plt.show()
            val = input("Press Enter if ok; \nType 'rm {int}' to remove a maxima; \nType 'add {int}' to add a maxima:\n")
            continue
        except Exception:
            print("Something went wrong >.<")
            print(sys.exc_info())
            val = input("Press Enter if ok; \nType 'rm {int}' to remove a maxima; \nType 'add {int}' to add a maxima:\n")
            continue

    f = plt.figure(figsize=(10, 4))
    # plt.title(file + ' Left')
    plt.plot(lJPos[0])
    plt.plot(data['leftWalkMode'])
    plt.plot(lMinimas, [lJPos[0][i] for i in lMinimas], 'r*')
    f.canvas.mpl_connect('button_press_event', onclick)
    plt.show()
    val = input("Press Enter if ok; \nType 'rm {int}' to remove a minima; \nType 'add {int}' to add a minima:\n")
    while val:
        try:
            plt.close()
            val = val.split(' ')
            if val[0] == 'rm':
                closest = min(lMinimas, key=lambda x : abs(x-int(val[1])))
                lMinimas.remove(closest)
            elif val[0] == 'add':
                lMinimas.append(int(val[1]))
                lMinimas.sort()
            else:
                print("Invalid Input!!!")
                raise Exception
            f = plt.figure(figsize=(10, 4))
            # plt.title(file + ' Left')
            plt.plot(lJPos[0])
            plt.plot(data['leftWalkMode'])
            plt.plot(lMinimas, [lJPos[0][i] for i in lMinimas], 'r*')
            f.canvas.mpl_connect('button_press_event', onclick)
            plt.show()
            val = input("Press Enter if ok; \nType 'rm {int}' to remove a minima; \nType 'add {int}' to add a minima:\n")
            continue
        except Exception:
            print("Something went wrong >.<")
            print(sys.exc_info())
            val = input("Press Enter if ok; \nType 'rm {int}' to remove a minima; \nType 'add {int}' to add a minima:\n")
            continue

    f = plt.figure(figsize=(10, 4))
    # plt.title(file + ' Right')
    plt.plot(rJPos[0])
    plt.plot(data['rightWalkMode'])
    plt.plot(rMinimas, [rJPos[0][i] for i in rMinimas], 'r*')
    f.canvas.mpl_connect('button_press_event', onclick)
    plt.show()

    val = input("Press Enter if ok; \nType 'rm {int}' to remove a minima; \nType 'add {int}' to add a minima:\n")
    while val:
        try:
            plt.close()
            val = val.split(' ')
            if val[0] == 'rm':
                closest = min(rMinimas, key=lambda x : abs(x-int(val[1])))
                rMinimas.remove(closest)
                print(f"Removed point {closest}")
            elif val[0] == 'add':
                rMinimas.append(int(val[1]))
                rMinimas.sort()
                print(f"Added point " + val[1])
            else:
                print("Invalid Input!!!")
                continue
            f = plt.figure(figsize=(10, 4))
            # plt.title(file + ' Right')
            plt.plot(rJPos[0])
            plt.plot(data['rightWalkMode'])
            plt.plot(rMinimas, [rJPos[0][i] for i in rMinimas], 'r*')
            f.canvas.mpl_connect('button_press_event', onclick)
            plt.show()
            val = input("Press Enter if ok; \nType 'rm {int}' to remove a minima; \nType 'add {int}' to add a minima:\n")
            continue
        except Exception:
            print("Something went wrong >.<")
            print(sys.exc_info())
            val = input("Press Enter if ok; \nType 'rm {int}' to remove a minima; \nType 'add {int}' to add a minima:\n")
            continue

    lY = pd.Series(np.nan, index=range(0, data.shape[0]))
    rY = pd.Series(np.nan, index=range(0, data.shape[0]))

    l_start_transition = np.diff(data['leftWalkMode']) == 4
    l_end_transition = np.diff(data['leftWalkMode']) == -4

    l_minima_bf = []
    for point in np.argwhere(l_start_transition):
        closest = min(lMinimas, key=lambda x : abs(x-point))
        if closest > point:
            closest = lMinimas[lMinimas.index(closest) - 1]
        l_minima_bf.append(closest)

    l_minima_af = []
    for point in np.argwhere(l_end_transition):
        closest = min(lMinimas, key=lambda x : abs(x-point))
        if closest < point:
            closest = lMinimas[lMinimas.index(closest) + 1]
        l_minima_af.append(closest)

    # plt.plot(data['leftWalkMode'])
    # plt.plot(data['leftJointPosition'])
    # plt.plot(minima_af, [data['leftJointPosition'][x] for x in minima_af], 'r*')
    # plt.plot(minima_bf, [data['leftJointPosition'][x] for x in minima_bf], 'y*')
    # plt.show()

    for start, end in zip(l_minima_bf, l_minima_af):
        lMaximas = [x for x in lMaximas if x < start or x > end]
        minimas = [x for x in lMinimas if x >= start and x <= end]
        diff = np.diff(minimas)
        for i in range(len(diff)):
            lMaximas.append(int(minimas[i]+diff[i]*0.6))
        lMaximas.sort()

    r_start_transition = np.diff(data['rightWalkMode']) == 4
    r_end_transition = np.diff(data['rightWalkMode']) == -4

    r_minima_bf = []
    for point in np.argwhere(r_start_transition):
        closest = min(rMinimas, key=lambda x : abs(x-point))
        if closest > point:
            closest = rMinimas[rMinimas.index(closest) - 1]
        r_minima_bf.append(closest)

    r_minima_af = []
    for point in np.argwhere(r_end_transition):
        closest = min(rMinimas, key=lambda x : abs(x-point))
        if closest < point:
            closest = rMinimas[rMinimas.index(closest) + 1]
        r_minima_af.append(closest)

    # plt.plot(data['leftWalkMode'])
    # plt.plot(data['leftJointPosition'])
    # plt.plot(minima_af, [data['leftJointPosition'][x] for x in minima_af], 'r*')
    # plt.plot(minima_bf, [data['leftJointPosition'][x] for x in minima_bf], 'y*')
    # plt.show()

    for start, end in zip(r_minima_bf, r_minima_af):
        rMaximas = [x for x in rMaximas if x < start or x > end]
        minimas = [x for x in rMinimas if x >= start and x <= end]
        diff = np.diff(minimas)
        for i in range(len(diff)):
            rMaximas.append(int(minimas[i]+diff[i]*0.6))
        rMaximas.sort()

    for maxima in lMaximas:
        lY[maxima] = 1
        lY[maxima+1] = 0
    for maxima in rMaximas:
        rY[maxima] = 1
        rY[maxima+1] = 0

    # Linearly interpolate the labels between every 0 and 1 and fill in the
    # rest with 0's
    # Conver to polar coordinates
    lY.interpolate(inplace=True), rY.interpolate(inplace=True)
    lY.fillna(0, inplace=True), rY.fillna(0, inplace=True)
    ly_theta, ry_theta = lY * 2 * np.pi, rY * 2 * np.pi
    left_x, left_y = np.cos(ly_theta), np.sin(ly_theta)
    right_x, right_y = np.cos(ry_theta), np.sin(ry_theta)
    labels = pd.DataFrame({'leftGaitPhaseX': left_x, 'leftGaitPhaseY': left_y,
                                'rightGaitPhaseX': right_x, 'rightGaitPhaseY': right_y})
    plt.plot(data['leftJointPosition'])
    plt.plot(lY)
    plt.show()

    # Combine the data and the labels
    data[labels.columns] = labels
    all_maximas = sorted(lMaximas + rMaximas)
    all_maximas = all_maximas[1:-1]

    if lMaximas[0]<rMaximas[0]:
        lMaximas = lMaximas[1:]
    else:
        rMaximas = rMaximas[1:]

    if lMaximas[-1]>rMaximas[-1]:
        lMaximas = lMaximas[:-1]
    else:
        rMaximas = rMaximas[:-1]

    data = data.iloc[all_maximas[0]:all_maximas[-1]+1, :]

    data.to_csv(filename)
    return data

def manual_scrap_data_eval(data, filename):
    """For each trial, plot the joint position and detected the peaks, when
    click on the graph, the x-coordinate of the mouse click will be printed and the
    plot will be closed. Then user need to enter "add {int: x-coord}" to add a
    peak, "rm {int: x-coord}" to remove the nearest peak, or press enter to move on
    to the next trial.
    A plot with both left and right joint will be displayed, when that plot is
    closed, a file containing the labeled data will be saved to the
    correspond subject's data folder.

    Args:
        subject (int): subject number
    """
    def onclick(event):
        print(event.xdata)
        plt.close()

    def plot_trial(lWalkmode, rWalkmode, lJPos, rJPos, start=False, end=False):
        f, axs = plt.subplots(2,1, sharex=True, figsize=(10, 4))
        axs[0].plot(lJPos[0])
        axs[0].plot(data['m_lCtrlTrq'])
        axs[0].plot(lWalkmode)
        axs[1].plot(rJPos[0])
        axs[1].plot(data['m_rCtrlTrq'])
        axs[1].plot(rWalkmode)
        if start != False:
            axs[0].axvline(start, color='r')
            axs[1].axvline(start, color='r')
        if end != False:
            axs[0].axvline(end, color='r')
            axs[1].axvline(end, color='r')
        axs[0].set_title('Left Hip Angle')
        axs[1].set_title('Right Hip Angle')
        f.canvas.mpl_connect('button_press_event', onclick)
        plt.show()

    lJPos, rJPos = extract_joint_positions([data])
    # lMaximas, rMaximas = find_local_maximas(lJPos[0]), find_local_maximas(rJPos[0])

    # Data for this file
    current_file_subsets = []

    # Plot graph, onclick -> print x coordinate and close graph
    # Get input from uesr to add peak, delete peak, or move on to the next plot

    print('Click anywhere to continue')
    plot_trial(data['leftWalkMode'], data['rightWalkMode'],lJPos, rJPos)
    val = input("Press 1 to keep all data\nPress 2 to discard all data\nPress 3 to input a starting point\n")
    if (val == "1"):
        # Save all data for this trial
        np.savetxt('chopped.txt')
    elif (val == "2"):
        print('Done with a trial!')
    else:
        # Repeat the following until 1 or 2 is pressed
        # - Show the current remaining data
        # - Ask for a starting point
        # - Ask for an ending point or press X to restart
        # - If restart, continue loop
        # - If ending point clicked, ask to confirm range of data by pressing enter or restart by pressing X
        # - If X, continue loop
        # - If enter, add start and end indices to current_file_subsets

        remaining_data = data
        while (val != "1" and val != "2"):
            print('Click to select a starting point')
            # Input a starting point
            plot_trial(remaining_data['leftWalkMode'], remaining_data['rightWalkMode'],lJPos, rJPos)
            val = input('Enter the starting point\n')
            start_ind = int(val)
            print('Click to select an ending point')
            plot_trial(remaining_data['leftWalkMode'], remaining_data['rightWalkMode'],lJPos, rJPos, start_ind)

            val = input('Enter the ending point or press x to restart\n')
            if (val == "x"):
                print('Restart trial')
                continue
            else:
                end_ind = int(val)
                print('Review the range. Click to respond')
                plot_trial(remaining_data['leftWalkMode'], remaining_data['rightWalkMode'],lJPos, rJPos, start_ind, end_ind)
                val = input('Confirm range by pressing enter or restart by pressing x\n')
                if (val == "x"):
                    print('Restart trial')
                    continue
                else:
                    current_file_subsets.append((start_ind, end_ind))
                    remaining_data = remaining_data.loc[end_ind:]
                    lJPos, rJPos = extract_joint_positions([remaining_data])
                    print('Click anywhere to continue')
                    plot_trial(remaining_data['leftWalkMode'], remaining_data['rightWalkMode'],lJPos, rJPos)
                    val = input("Press 1 to keep all data\nPress 2 to discard all data\nPress 3 to input a starting point\n")

        for i, (start, end) in enumerate(current_file_subsets, start=1):
            clip = data.loc[start:end]
            # Save all data for this trial
            clip.to_csv(filename)

def manual_segment_magnitudes(data, filename):
    def onclick(event):
        print(event.xdata)

    def plot_trial(lJPos, rJPos, start=False, end=False):
        f, axs = plt.subplots(2,1, sharex=True, figsize=(10, 4))
        f.suptitle(file)
        axs[0].plot(lJPos[0])
        axs[1].plot(rJPos[0])
        if start != False:
            axs[0].axvline(start, color='r')
            axs[1].axvline(start, color='r')
        if end != False:
            axs[0].axvline(end, color='r')
            axs[1].axvline(end, color='r')
        axs[0].set_title('Left Hip Angle')
        axs[1].set_title('Right Hip Angle')
        f.canvas.mpl_connect('button_press_event', onclick)
        plt.show()

    data['mag'] = -1
    
    plot_data(data)
    val = input("Press Enter to set mag, press anything else to finish")
    
    while val == "":
        start = input("Start Index: ")
        end = input("End Index: ")
        mag = input("Magnitude: ")
        data['mag'].iloc[int(start):int(end)] = float(mag)
        plot_data(data)
        val = input("Press Enter to set mag, press anything else to finish")
        
    data.to_csv(filename)
    return data