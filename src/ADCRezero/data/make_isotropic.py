import numpy as np
from scipy.signal import istft, get_window
from scipy.special import j0
import soundfile as sf

def coherence_matrix(freqs, mic_positions, c=343.0, field='3d'):
    """
    Returns the spatial coherence matrix Γ(f) for each frequency.
    
    input:
        freqs: (K,)  rFFT周波数 [Hz]
        mic_positions: (M, 3) [m]
        field: '3d' -> sinc, '2d' -> J0
        return: (K, M, M) complex64
    """
    M = mic_positions.shape[0]
    diff = mic_positions[:, None, :] - mic_positions[None, :, :]
    dists = np.linalg.norm(diff, axis=-1)  # (M, M)

    K = len(freqs)
    Gamma = np.zeros((K, M, M), dtype=np.complex64)
    np.fill_diagonal(dists, 0.0)
    
    for k, f in enumerate(freqs):
        if field == '3d':
            x = 2.0 * np.pi * f * dists / c
            with np.errstate(invalid='ignore', divide='ignore'):
                g = np.ones_like(x)
                nz = (x != 0)
                g[nz] = np.sin(x[nz]) / x[nz]
        elif field == '2d':
            x = 2.0 * np.pi * f * dists / c
            g = j0(x)
        else:
            raise ValueError("field must be '3d' or '2d'")
        
        np.fill_diagonal(g, 1.0)
        Gamma[k] = g.astype(np.complex64)
    return Gamma

def stable_matrix_sqrt_hermitian(R, reg=1e-9):
    """
    Returns the stable square root matrix of Hermitian positive definite (semi)matrix R.
    Perform eigenvalue decomposition using eigh, clip negative eigenvalues to zero, take the square root, and return the result.
    
    input:
        R: (M, M) Hermitian
        reg: regularization term added to the diagonal
    return: 
        (M, M) so that R ≈ L @ L^H
    """
    Rh = 0.5 * (R + R.conj().T)
    M = Rh.shape[0]
    Rh = Rh + reg * np.eye(M, dtype=Rh.dtype)
    w, U = np.linalg.eigh(Rh)
    cond_num = np.max(w) / (np.min(w) + 1e-15)
    if cond_num > 1e12:
        print(f"Warning: High condition number {cond_num:.2e}, may affect accuracy")
    
    w = np.clip(w, 0.0, None)
    sqrt_w = np.sqrt(w)
    L = (U * sqrt_w[None, :]) @ U.conj().T   # L = U * sqrt(D) * U^H (Cholesky-like decomposition)
    return L

def psd_shaping(freqs, kind=None, f_ref=1000.0):
    """
    Returns the square root of |S(f)| (amplitude spectrum).
    
    input:
        freqs: (K,) rFFT周波数 [Hz]
        kind: 'white' or 'pink', if None, randomly choose
        f_ref: reference frequency for normalization (not used in current implementation)
    return:
        (K,) float64
    """
    if kind is None:
        kind = np.random.choice(['white', 'pink'])
    amp = np.ones_like(freqs, dtype=np.float64)
    if kind == 'pink':
        amp[1:] = 1.0 / np.sqrt(np.maximum(freqs[1:], 1.0))
        amp[0] = amp[1]
    return amp

def generate_diffuse_noise(mic_pos, sr=16000, duration_sec=6.0, field='3d', n_fft=32, hop_length=8, window='hann', 
                            psd_kind=None, target_rms=None, rng=None, reg=1e-9, c=343.0):
    """
    Generate M-channel isotropic (diffuse) noise.
    
    input:
        mic_positions: (M, 2) or (M, 3) [m]
        sr: sampling rate [Hz]
        duration_sec: duration of the noise [sec]
        field: '3d' (sinc) or '2d' (J0)
        n_fft: FFT size for STFT
        hop_length: hop length for STFT
        window: window type for STFT
        psd_kind: 'white' or 'pink', if None, randomly choose
        target_rms: if not None, normalize to this RMS level
        rng: numpy random generator, if None, create a new one
        reg: regularization for matrix square root
        c: speed of sound [m/s]
    return:
        noise: (M, T) float32
    """
    mic_pos = np.asarray(mic_pos, dtype=np.float64)
    assert mic_pos.ndim == 2 and mic_pos.shape[1] in (2, 3)
    if mic_pos.shape[1] == 2:
        mic_pos = np.pad(mic_pos, ((0, 0), (0, 1)), mode='constant', constant_values=0.0)
    M = mic_pos.shape[0]

    if rng is None:
        rng = np.random.default_rng()

    T = int(round(duration_sec * sr))
    nperseg = n_fft
    noverlap = n_fft - hop_length
    win = get_window(window, nperseg, fftbins=True)

    # rFFT frequencies
    freqs = np.fft.rfftfreq(nperseg, d=1.0/sr)
    K = len(freqs)

    # Coherence matrix Γ(f) for each frequency
    Gamma = coherence_matrix(freqs, mic_pos, c=c, field=field)  # (K, M, M)

    # PSD Shaping (Amplitude)
    amp = psd_shaping(freqs, kind=psd_kind).astype(np.float64)        # (K,)
    
    n_frames = max(1, int(np.ceil((T - nperseg) / (nperseg - noverlap))) + 1)
    X = np.zeros((M, K, n_frames), dtype=np.complex64)
    for k in range(K):
        R = Gamma[k] * (amp[k] ** 2)                                  # (M, M)
        L = stable_matrix_sqrt_hermitian(R, reg=reg)                  # (M, M)

        # Prepare n_frames independent complex Gaussian CN(0, I)
        z_real = rng.standard_normal((M, n_frames))
        z_imag = rng.standard_normal((M, n_frames))
        Z = (z_real + 1j * z_imag).astype(np.complex64) / np.sqrt(2.0)
        Xk = (L @ Z).astype(np.complex64)                             # (M, n_frames)
        
        # Proper Handling of DC Components and Nyquist Components
        if k == 0:
            # DC component: Real number and maintains spatial correlation
            Xk = Xk.real.astype(np.complex64)
        elif nperseg % 2 == 0 and k == K - 1:
            # Nyquist component: real-valued correlated component
            z_real_ny = rng.standard_normal((M, n_frames))
            Xk_real = (L.real @ z_real_ny).astype(np.float32)
            Xk = Xk_real.astype(np.complex64)
        X[:, k, :] = Xk
    
    noise = []
    for m in range(M):
        _, xm = istft(X[m], fs=sr, window=win, nperseg=nperseg, noverlap=noverlap, input_onesided=True, boundary=True)
        if len(xm) < T:
            pad = np.zeros(T - len(xm), dtype=np.float32)
            xm = np.concatenate([xm.astype(np.float32), pad], axis=0)
        else:
            xm = xm[:T].astype(np.float32)
        noise.append(xm)
    noise = np.stack(noise, axis=0)                                    # (M, T)
    
    if target_rms is not None and target_rms > 0:
        channel_rms = np.sqrt(np.mean(noise**2, axis=1, keepdims=True))
        overall_rms = np.sqrt(np.mean(channel_rms**2))
        
        if overall_rms > 0:
            scale_factor = target_rms / overall_rms
            noise *= scale_factor
            target_channel_rms = target_rms / np.sqrt(M)
            curr_channel_rms = np.sqrt(np.mean(noise**2, axis=1, keepdims=True))
            with np.errstate(divide='ignore', invalid='ignore'):
                per_ch_scale = np.where(curr_channel_rms > 0, target_channel_rms / curr_channel_rms, 1.0)
            noise *= per_ch_scale
            if np.std(channel_rms.flatten()) / np.mean(channel_rms.flatten()) > 0.1:
                print("Warning: Channel energy imbalance detected in diffuse noise")
    return noise # (M, T)

if __name__ == "__main__":
    M = 8
    radius = 0.025  # 2.5 cm
    angles = np.linspace(0, 2*np.pi, M, endpoint=False)
    mic_xy = np.stack([radius * np.cos(angles), radius * np.sin(angles)], axis=1)  # (M,2)
    sr = 16000
    out_path = "outputs/diffuse_noise.wav"
    
    noise = generate_diffuse_noise(mic_pos=mic_xy, sr=sr, duration_sec=6.0, field='3d', n_fft=32, hop_length=8, window='hann', psd_kind='pink',)
    sf.write(out_path, noise.T, sr)
    print(f"Wrote diffuse noise to {out_path}")