import numpy as np

def convert_pvalue_to_asterisks(pvalue):
    if pvalue <= 0.001:
        return "***"
    elif pvalue <= 0.01:
        return "**"
    elif pvalue <= 0.05:
        return "*"
    return ""

def batch_cosine_similarity(a, b, dim=-1):
    return (a/(np.linalg.norm(a, axis=dim))[...,None]) @ np.swapaxes(b/(np.linalg.norm(b, axis=dim))[...,None], -1, -2)

def get_sub_mats(ws, num_areas, e_hidden_size, i_hidden_size, separate_ei=True):
    trials, timesteps, batch_size, post_dim, pre_dim = ws.shape
    assert((e_hidden_size+i_hidden_size)*num_areas==pre_dim and (e_hidden_size+i_hidden_size)*num_areas==post_dim)
    total_e_size = e_hidden_size*num_areas
    submats = {}
    if not separate_ei:
        for i in range(num_areas):
            submats[f"rec_intra_{i}"] = ws[:,:,:,list(range(i*e_hidden_size, (i+1)*e_hidden_size))+\
                                                list(range(total_e_size+i*i_hidden_size, total_e_size+(i+1)*i_hidden_size))]\
                                        [:,:,:,:,list(range(i*e_hidden_size, (i+1)*e_hidden_size))+\
                                                list(range(total_e_size+i*i_hidden_size, total_e_size+(i+1)*i_hidden_size))]

        for i in range(num_areas-1):
            submats[f"rec_inter_ff_{i}_{i+1}"] = ws[:,:,:,list(range((i+1)*e_hidden_size, (i+2)*e_hidden_size))+\
                                                    list(range(total_e_size+(i+1)*i_hidden_size, total_e_size+(i+2)*i_hidden_size))]\
                                            [:,:,:,:,list(range(i*e_hidden_size, (i+1)*e_hidden_size))]
            submats[f"rec_inter_fb_{i+1}_{i}"] = ws[:,:,:,list(range(i*e_hidden_size, i*e_hidden_size))+\
                                                    list(range(total_e_size+(i+1)*i_hidden_size, total_e_size+(i+2)*i_hidden_size))]\
                                            [:,:,:,:,list(range((i+1)*e_hidden_size, (i+2)*e_hidden_size))]
        return submats
    else:
        for i in range(num_areas):
            e_indices = list(range(i*e_hidden_size, (i+1)*e_hidden_size))
            i_indices = list(range(total_e_size+i*i_hidden_size, total_e_size+(i+1)*i_hidden_size))

            submats[f"rec_intra_ee_{i}"] = ws[...,e_indices,:][:,:,:,:,e_indices]
            submats[f"rec_intra_ie_{i}"] = ws[...,i_indices,:][:,:,:,:,e_indices]
            submats[f"rec_intra_ei_{i}"] = ws[...,e_indices,:][:,:,:,:,i_indices]
            submats[f"rec_intra_ii_{i}"] = ws[...,i_indices,:][:,:,:,:,i_indices]

        for i in range(num_areas-1):
            e_hi_indices = list(range((i+1)*e_hidden_size, (i+2)*e_hidden_size))
            e_lo_indices = list(range(i*e_hidden_size, (i+1)*e_hidden_size))
            i_hi_indices = list(range(total_e_size+(i+1)*i_hidden_size, total_e_size+(i+2)*i_hidden_size))
            i_lo_indices = list(range(total_e_size+i*i_hidden_size, total_e_size+(i+1)*i_hidden_size))

            submats[f"rec_inter_ff_ee_{i}_{i+1}"] = ws[:,:,:,e_hi_indices,:][:,:,:,:,e_lo_indices]
            submats[f"rec_inter_ff_ie_{i}_{i+1}"] = ws[:,:,:,i_hi_indices,:][:,:,:,:,e_lo_indices]
            submats[f"rec_inter_fb_ee_{i+1}_{i}"] = ws[:,:,:,e_lo_indices,:][:,:,:,:,e_hi_indices]
            submats[f"rec_inter_fb_ie_{i+1}_{i}"] = ws[:,:,:,i_lo_indices,:][:,:,:,:,e_hi_indices]
        
        return submats

def plot_mean_and_std(ax, m, sd, label, color, alpha=1):
    if label is not None:
        ax.plot(m, alpha=alpha, label=label, c=color)
    else:
        ax.plot(m, alpha=alpha, c=color)
    ax.fill_between(range(len(m)), m-sd, m+sd, color=color, alpha=0.1)

def plot_imag_centered_cm(ax, im):
    max_mag = im.abs().max()*0.3
    im = ax.imshow(im, vmax=max_mag, vmin=-max_mag, cmap='RdBu_r')
    return im
