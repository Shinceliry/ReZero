import numpy as np
import torch
from pathlib import Path
import argparse
from tqdm import tqdm
from src.ADCRezero.data.onthefly import RezeroOnTheFlyDataset

class RezeroDatasetGenerator(RezeroOnTheFlyDataset):
    def __init__(self, args):
        '''
        Addtional arguments:
        args.iterations (int): Number of iterations for the dataset.    
        args.output_dir (str): Directory to save the generated data.
        args.plot_3d (bool): Whether to plot 3D simulation setup.
        '''
        super().__init__(args)

    def __len__(self):
        return self.args.iterations

    def __getitem__(self, idx):
        data = super().__getitem__(idx)
        mix = data['mix']
        theta_l_query = data['theta_l_query'].tolist()
        theta_h_query = data['theta_h_query'].tolist()
        d_query = data['d_query'].tolist()
        selected_speech_wavs = data['selected_speech_wavs']
        selected_noise_wavs = data['selected_noise_wavs']
        speech_pos = data['speech_pos'].tolist()
        noise_pos = data['noise_pos']
        target = data['target']
        region_indices = data['region_indices'].tolist()
        q_a = data['q_a'].item()
        q_d = data['q_d'].item()
        use_diffuse = data['use_diffuse']
        diffuse_noise = data['diffuse_noise']
        n_n_wo_diffuse_noise = data['n_n_wo_diffuse_noise']
        
        B = mix.size(0)
        speech_pos = np.array(speech_pos, dtype=np.float32)
        
        # save metadata and audio
        out_root = Path(self.args.output_dir) / f"{idx:06d}"
        out_root.mkdir(parents=True, exist_ok=True)
        self.save_metadata_and_audio(out_root, theta_l_query, theta_h_query, d_query, 
                            speech_pos, noise_pos, self.array_pos[0], region_indices, q_a, q_d,
                            selected_speech_wavs, selected_noise_wavs, mix, target, use_diffuse, diffuse_noise, n_n_wo_diffuse_noise)
        
        # Plot 3D simulation
        if self.args.plot_3d and (self.args.mode == 'val' or self.args.mode == 'test'):
            self.simulation_plot_3d(out_root, speech_pos, noise_pos, self.array_pos[0], theta_l_query, theta_h_query, d_query, region_indices)
        
        print(f"Generated mixture {idx+1}/{self.args.iterations}, Q={self.Q_val}")
        for attr in ('n_s', 'n_n', 'Q_val', 'room', 'rt60', 'mic_pos', 'array_pos'):
            if hasattr(self, attr):
                delattr(self, attr)
        
        return {
            'ok': True,
        }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate test mixtures with metadata using RezeroTestset")
    parser.add_argument("--speech_dir", type=str, required=True, help="Path to directory of speech files")
    parser.add_argument("--noise_dir", type=str, required=True, help="Path to directory of noise files")
    parser.add_argument("--output_dir", type=str, required=True, help="Where to save mix/target WAVs and metadata")
    parser.add_argument('--config', default='src/ADCRezero/config/original.yaml', help='Path to config.yaml')
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size (number of mixtures per data)")
    parser.add_argument("--num_workers", type=int, default=16, help="Number of subprocesses for data loading")
    parser.add_argument("--iterations", type=int, default=3000, help="Number of testset")
    parser.add_argument("--mode", type=str, required=True, help="train or val or test")
    parser.add_argument("--mic_arch", choices=["circular", "linear"], default="circular", help="Microphone array architecture")
    parser.add_argument("--region_type", choices=["angular", "spherical", "conical"], default="conical", help="Type of query region to sample")
    parser.add_argument("--first_positioning", choices=["mic", "speaker"], default="mic", help="First positioning of sources")
    parser.add_argument("--decision_query_region", choices=['no_limit', 'angle_limit', 'distance_limit'], help="Method for deciding query region if args.first_positioning is 'mic'")
    parser.add_argument("--limit_mic_z", action='store_true', help="Limit microphone z-coordinate to a specific range")
    parser.add_argument("--elevation_limit", action='store_true', help="Limit elevation of sources to a specific range")
    parser.add_argument("--mic_in_center", action='store_true', help="Place the microphone array at the center of the room")
    parser.add_argument("--room_size", type=float, nargs=3, default=None, help="Room size in meters (x y z)")
    parser.add_argument("--infinity_room", action='store_true', help="Use an infinite room for simulation (no reverberation)")
    parser.add_argument("--query2D", action='store_true', help="Consider the query area in a two-dimensional plane.")
    parser.add_argument("--spherical_fixed_azimuth", action='store_true', help="For spherical region type, fix azimuth angles when sampling sources.")
    parser.add_argument("--no_noise", action='store_true', help="Exclude noise sources from the mixtures")
    parser.add_argument("--diffuse_noise_prob", type=float, default=0.1, help="Probability of using diffuse noise when n_n > 0")
    parser.add_argument("--rms_norm", action="store_true", help="Apply RMS normalization to the mixture")
    parser.add_argument("--plot_3d", action='store_true', help="Plot 3D simulation setup")
    parser.add_argument("--one_speaker", action='store_true', help="Use only one speaker in the simulation")
    parser.add_argument("--cpuram", action='store_true', help="Load audio files into RAM")
    args = parser.parse_args()
    args.dataset_generation = True
    
    dataset = RezeroDatasetGenerator(args)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=False,
        persistent_workers=False,
    )
    
    print(f"Dataset length (iterations): {len(dataset)}")
    for i, _ in tqdm(enumerate(dataloader)):
        if i + 1 >= len(dataset):
            break