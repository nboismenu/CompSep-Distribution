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

def iPS(L):
    image = np.zeros((N,N))
    creneau = np.meshgrid(np.linspace(-N//2+1, N//2, N), np.linspace(-N//2+1, N//2, N))
    T = np.sqrt(creneau[0]**2 + creneau[1]**2)
    kbins = np.arange(0, len(L) + 2)
    for i in range(len(L)):
        mask = np.bitwise_and(kbins[i]<=T, T<kbins[i+1])
        image[mask] = np.sqrt(L[i])    
    mask = T>kbins[i+1]
    image[mask] = L[i]
    imR = np.random.randn(N, N)
    fourier = np.fft.fftshift(image) * np.exp(1j * np.angle(np.fft.fft2(imR)))
    return np.real(np.fft.ifft2(fourier))

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
nb_bruit = 20 # nb of noise map
noise = noise_std_ratio*np.std(x)*np.random.randn(nb_bruit, np.shape(x)[0], np.shape(x)[1])

d_coeffs = wph_op.apply(d, pbc = pbc)

nb_synt = 50


def lissage(L):
    return (L[2:] + L[1:-1] + L[:-2])/3
from_PS_list = []
for i in range(nb_synt):
    rd = np.random.uniform(-0.99,1,len(PS(x))+6)
    rd = lissage(rd)
    rd[10:-2] = lissage(rd[10:])
    rd = rd[:-2]
    rd[50:-2] = lissage(rd[50:])
    rd = rd[:-2]
    rd = rd * (PS(noise[0])/PS(x))**0.3
    from_PS = PS(x) *(10**rd)
    from_PS[from_PS<0] = 0
    from_PS_list.append(from_PS)

d_list = np.array([iPS(from_PS) for from_PS in from_PS_list])
std = compute_std(5)
loss_graph = []
d = d_list[:]

for k in range(len(d_list)):
    d_c = d[k]
    for i in range(3):
        loss_current = []
        d_c = d_c.reshape(M, N)
        coefs_u_bruits = [wph_op.apply(d_c + n, pbc = pbc) for n in noise]
        biais = 1/nb_bruit*sum(coefs_u_bruits) - wph_op.apply(d_c, pbc = pbc)

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
        result = opt.minimize(objective_JM, d_c.ravel(), method='L-BFGS-B', jac=True, tol=None, options={"maxiter": 60, "gtol": 1e-14, "ftol": 1e-14, "maxcor": 20})
        final_loss, d_c, niter, msg = result['fun'], result['x'], result['nit'], result['message']
        d_c = d_c.reshape(M, N)
        loss_graph.append(loss_current)
        print(i, final_loss)
    d[k] = d_c[:]
#sep_from_WN = d[:]

plt.figure()
k = np.arange(0, N//2 + 1)
n_PS = np.array([PS(n) for n in noise])
plt.loglog(k, PS(d_init), color = 'blue', label = 'image bruitée (d_init)')
plt.loglog(k, PS(x), label = 'image non bruitée (x)', color = 'purple')
for d_c in d:
    plt.loglog(k, PS(d_c), color = 'red', alpha = 0.4)

plt.loglog(k, np.mean(n_PS, axis=0), label = 'Bruit mean et 3std (n)', color = 'green')
plt.fill_between(k, np.mean(n_PS, axis=0)- 1*np.std(n_PS, axis=0), np.mean(n_PS, axis=0) + 1*np.std(n_PS, axis=0), color = 'green', alpha = 0.3)

plt.legend()
plt.savefig('_famille_PS')

np.save('_syntheses.npy', d)