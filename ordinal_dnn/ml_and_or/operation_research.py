import numpy as np
from ortools.linear_solver import pywraplp


def sum_rows(R, num_of_samples, num_of_labels):
    # G = np.zeros(num_of_labels)
    G = []
    for j in range(num_of_labels):
        label_sum = 0
        count=0
        for i in range(num_of_samples):
            # G[i] += R[(i, j)]
            label_sum += R[(i, j)]
            # count += R[(i, j)].solution_value()
        G.append(label_sum)
        # print(count)
    return G


def sum_columns(R, num_of_samples, num_of_labels):
    # G = np.zeros(num_of_labels)
    S = []
    for i in range(num_of_samples):
        row_sum = 0
        for j in range(num_of_labels):
            row_sum += R[(i, j)]
        S.append(row_sum)
    return S


def fault_checks_amount(num_of_labels, const_num, labels_for_const):
    n = np.ones(num_of_labels)
    # for j in range(num_of_labels):

    for j in labels_for_const:
        n[j] = const_num/100  # percentage for each fault
    return n


def operation_research_func(predict_labels_proba, num_of_labels, cost_matrix, fault_price, const_num, labels_for_const):
    R = {}
    objective_function = 0
    num_of_samples = predict_labels_proba.shape[0]

    predict_hard = np.zeros(num_of_samples).astype(int)

    np.random.seed(42)
    solver = pywraplp.Solver.CreateSolver('GLOP')
    # print('probe:', predict_training_labels_proba)
    cost_dot_proba = np.dot(cost_matrix, predict_labels_proba.transpose())
    # print('cost shape:')
    # print(cost_matrix.shape)
    # print('proba shape:')
    # print(predict_training_labels_proba.shape)
    # print('C * Y^T:')
    # print(cost_dot_proba.shape)
    for i in range(num_of_samples):
        for j in range(num_of_labels):
            # R[(i, j)] = solver.IntVar(0, solver.infinity(), 'R{}{}'.format(i, j))
            R[(i, j)] = solver.NumVar(0, solver.infinity(), 'R{}{}'.format(i, j))
            objective_function += R[(i, j)] * cost_dot_proba[j, i] + R[(i, j)] * fault_price[j]
    # print('Number of variables =', solver.NumVariables())
    # Objective function
    solver.Minimize(objective_function)

    # Constraints

    n = fault_checks_amount(num_of_labels, const_num, labels_for_const)
    G = sum_rows(R, num_of_samples, num_of_labels)
    for i in range(num_of_labels):
        # print(np.ceil(num_of_samples * n[1]))
        solver.Add(G[i] <= np.ceil(num_of_samples * n[i]))

    # print('const class 2')
    # print(np.ceil(num_of_samples * n[2]))
    # solver.Add(G[2] <= np.ceil(num_of_samples * n[2]))

    S = sum_columns(R, num_of_samples, num_of_labels)
    for j in range(num_of_samples):
        solver.Add(S[j] == 1)

    # print('Number of constraints =', solver.NumConstraints())


    status = solver.Solve()

    assert status == pywraplp.Solver.OPTIMAL, 'The problem does not have an optimal solution.'
    # print('Solution:')
    objective_function_value = solver.Objective().Value()
    # print('Objective value:' + str(objective_function_value))
    # print('Objective value, NumVar=', objective_function_value)
    for i in range(num_of_samples):
        save = []
        row_place = np.argmax(predict_labels_proba[i, :])
        column_place = -1
        for j in range(num_of_labels):
            # print(' R[({}, {})] ='.format(i, j),  R[(i, j)].solution_value())
            save.append(R[(i, j)].solution_value())
            # print(R[(i, j)].solution_value())
            if R[(i, j)].solution_value() == 1:
                column_place = j
                # print('row & coulmn {}, {}'.format(row_place,j))
                predict_hard[i] = j
        # assert column_place != -1, 'cannot find column for row={}'.format(i)

    return objective_function_value, predict_hard