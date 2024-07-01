import numpy as np
import matplotlib.pyplot as plt
import pywph as pw
import torch
import time
import scipy.optimize as opt

def compute_std(nb_noise):
    A = torch.zeros(nb_noise, x_coeffs.shape[0])
    for i in range(nb_noise):
        A[i] = wph_op.apply(x + noise[i], pbc = pbc)
    std = torch.std(A, axis = 0)
    std = std.to('cuda')
    return std


def PS(image):
    ampl_TF = np.abs(np.fft.fft2(image))**2
    ampl_TF = np.fft.fftshift(ampl_TF)
    L = np.meshgrid(np.linspace(-N//2+1, N//2, N), np.linspace(-N//2+1, N//2, N))
    T = np.sqrt(L[0]**2 + L[1]**2)
    kbins = np.arange(0, N//2 + 2)
    PS = np.array([np.mean(ampl_TF[np.bitwise_and(kbins[i]<=T, T<kbins[i+1])]) for i in range(N//2+1)])
    return PS


pbc = True
x = np.load("Turb_3_MHD.npy")[5]
x = (x - x.mean())/x.std()

M, N = x.shape
J = int(np.log2(N)-2)
L = 4
dn = 0

wph_op = pw.WPHOp(M, N, J, L=L, dn=5, device=0)
x_coeffs = wph_op.apply(x, pbc = pbc)

noise_std_ratio = 1
d = x + noise_std_ratio*np.std(x)*np.random.randn(np.shape(x)[0], np.shape(x)[1])

d_init = d[:]
nb_bruit = 50 # nb of noise map
noise = noise_std_ratio*np.std(x)*np.random.randn(nb_bruit, np.shape(x)[0], np.shape(x)[1])
d.shape

d_coeffs = wph_op.apply(d, pbc = pbc)
std = compute_std(5)
loss_graph = []

########################## 1st CompSep

d = np.std(d)*np.random.randn(np.shape(x)[0], np.shape(x)[1])

for i in range(3):
    loss_current = []
    d = d.reshape(M, N)
    coefs_u_bruits = [wph_op.apply(d+ n, pbc = pbc) for n in noise]
    biais = 1/nb_bruit*sum(coefs_u_bruits) - wph_op.apply(d, pbc = pbc)

    def objective_JM(y):
        global eval_cnt
        #print(f"Evaluation: {eval_cnt}")
        start_time = time.time()

        # Reshape y
        y_curr = y.reshape(M, N)

        # Compute the loss (squared 2-norm)
        loss_tot = torch.zeros(1)
        y_curr, nb_chunks = wph_op.preconfigure(y_curr, requires_grad=True)
        for i in range(nb_chunks):
            y_coeffs_chunk, indices = wph_op.apply(y_curr, i, ret_indices=True, pbc = pbc)
            #print(indices.shape, y_coeffs_chunk.shape)
            loss = torch.sum(torch.abs((d_coeffs[indices] - y_coeffs_chunk - biais[indices])/std[indices]) ** 2)
            loss.backward(retain_graph=True)
            loss_tot += loss.detach().cpu()
            del y_coeffs_chunk, indices, loss

        # Reshape the gradient
        y_grad = y_curr.grad.cpu().numpy().astype(y.dtype)
        loss_current.append(loss_tot.item())
        #print(f"Loss: {loss_tot.item()} (computed in {(time.time() - start_time):.2f}s)")
        eval_cnt += 1
        return loss_tot.item(), y_grad.ravel()

    eval_cnt = 0
    result = opt.minimize(objective_JM, d.ravel(), method='L-BFGS-B', jac=True, tol=None, options={"maxiter": 100, "gtol": 1e-14, "ftol": 1e-14, "maxcor": 20})
    final_loss, d, niter, msg = result['fun'], result['x'], result['nit'], result['message']
    d = d.reshape(M, N)
    print(i, final_loss)
    loss_graph.append(loss_current)
s = d[:]


##################### CompSep on s

nb_synthèses = 50
if nb_synthèses > nb_bruit:
    raise TypeError('Le nb de synthèse doit être inférieur au nb de bruits')


d = np.random.randn(nb_synthèses, np.shape(x)[0], np.shape(x)[1])
loss_graph = []

for k in range(nb_synthèses):
    d_c = d[k]
    s_plus_n_coeffs = wph_op.apply(s + noise[k], pbc = pbc)
    for i in range(2):
        loss_current = []
        d_c = d_c.reshape(M, N)
        coefs_u_bruits = [wph_op.apply(d_c + n, pbc = pbc) for n in noise]
        biais = 1/nb_bruit*sum(coefs_u_bruits) - wph_op.apply(d_c, pbc = pbc) ############### PB taille

        def objective_JM(y):
            global eval_cnt
            #print(f"Evaluation: {eval_cnt}")
            start_time = time.time()

            # Reshape y
            y_curr = y.reshape(M, N)

            # Compute the loss (squared 2-norm)
            loss_tot = torch.zeros(1)
            y_curr, nb_chunks = wph_op.preconfigure(y_curr, requires_grad=True)
            for i in range(nb_chunks):
                y_coeffs_chunk, indices = wph_op.apply(y_curr, i, ret_indices=True, pbc = pbc)
                #print(indices.shape, y_coeffs_chunk.shape)
                loss = torch.sum(torch.abs((s_plus_n_coeffs[indices] - y_coeffs_chunk - biais[indices])/std[indices]) ** 2)
                loss.backward(retain_graph=True)
                loss_tot += loss.detach().cpu()
                del y_coeffs_chunk, indices, loss

            # Reshape the gradient
            y_grad = y_curr.grad.cpu().numpy().astype(y.dtype)
            loss_current.append(loss_tot.item())
            #print(f"Loss: {loss_tot.item()} (computed in {(time.time() - start_time):.2f}s)")
            eval_cnt += 1
            return loss_tot.item(), y_grad.ravel()

        eval_cnt = 0
        result = opt.minimize(objective_JM, d_c.ravel(), method='L-BFGS-B', jac=True, tol=None, options={"maxiter": 50, "gtol": 1e-14, "ftol": 1e-14, "maxcor": 20})
        final_loss, d_c, niter, msg = result['fun'], result['x'], result['nit'], result['message']
        d_c = d_c.reshape(M, N)
        loss_graph.append(loss_current)
        print(k, i, final_loss)
    d[k] = d_c[:]
sep_from_WN = d[:]

fig, axs = plt.subplots(2, 2, figsize=(9, 9), sharex=True, sharey=True)
vmin=x.mean() - 3*x.std()
vmax=x.mean() + 3*x.std()
axs[0,0].imshow(x, vmin=vmin, vmax=vmax)
axs[0,1].imshow(s, vmin=vmin, vmax=vmax)
axs[1,0].imshow(d[0], vmin=vmin, vmax=vmax)
axs[1,1].imshow(d[1], vmin=vmin, vmax=vmax)
fig.tight_layout()
fig.savefig('_Syntheses_1_et_2')


plt.figure()
k = np.arange(0, N//2 + 1)
plt.figure()
n_PS = np.array([PS(n) for n in noise])
plt.loglog(k, PS(d_init), color = 'blue', label = 'image bruitée (d_init)')
plt.loglog(k, PS(x), label = 'image non bruitée (x)', color = 'purple')
plt.loglog(k, PS(d[0]), color = 'red', alpha = 0.5, label='Ensemble des synt à vers s + n[k]')
for d_c in d[1:]:
    plt.loglog(k, PS(d_c), color = 'red', alpha = 0.5)

plt.loglog(k, PS(s), label = '1ere synthèse (s)', color = 'black')

plt.loglog(k, np.mean(n_PS, axis=0), label = 'Bruit mean et 3std (n)', color = 'green')
plt.fill_between(k, np.mean(n_PS, axis=0)- 1*np.std(n_PS, axis=0), np.mean(n_PS, axis=0) + 1*np.std(n_PS, axis=0), color = 'green', alpha = 0.5)
plt.legend()
plt.savefig('_famille_PS')

np.save('_syntheses.npy', d)