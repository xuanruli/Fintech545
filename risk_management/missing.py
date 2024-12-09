import numpy as np

def missing_cov(data, skipMiss=True, fun=np.cov):
    n, m = data.shape
    result_matrix = np.zeros((m, m))

    if skipMiss:
        # skip row
        global_valid_idx = np.all(~np.isnan(data), axis=1)
        clean_data = data[global_valid_idx, :]
        for i in range(m):
            for j in range(m):
                subset_data = clean_data[:, [i, j]]
                if subset_data.shape[0] > 1:
                    mat = fun(subset_data, rowvar=False)
                    result_matrix[i,j] = mat[0,1]
                else:
                    result_matrix[i,j] = np.nan
    else:
        # pairwise
        for i in range(m):
            for j in range(m):
                valid_idx = ~np.isnan(data[:,i]) & ~np.isnan(data[:,j])
                subset_data = data[valid_idx][:, [i,j]]
                if subset_data.shape[0] > 1:
                    mat = fun(subset_data, rowvar=False)
                    result_matrix[i,j] = mat[0,1]
                else:
                    result_matrix[i,j] = np.nan

    return result_matrix