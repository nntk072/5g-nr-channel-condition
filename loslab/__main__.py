"""CLI: python -m loslab {doctor,debug-data,phy-data,rt-data,experiment}."""
from __future__ import annotations
import argparse,json
from pathlib import Path


def main() -> None:
    parser=argparse.ArgumentParser(prog='python -m loslab')
    sub=parser.add_subparsers(dest='command',required=True)
    sub.add_parser('doctor',help='Report installed Sionna/PyTorch versions, no simulation run')
    city=sub.add_parser('city-scene',help='Generate reproducible 3D urban scene files')
    city.add_argument('--output-dir',required=True)
    city.add_argument('--seed',type=int,default=42)
    city.add_argument('--grid-size',type=int,default=5)
    city.add_argument('--vehicles',type=int,default=12)
    demo=sub.add_parser('debug-data',help='Generate synthetic/debug smoke-test data, NOT results')
    demo.add_argument('--output',required=True);demo.add_argument('--seed',type=int,default=42)
    demo.add_argument('--scenes',type=int,default=5)
    demo.add_argument('--links-per-scene',type=int,default=32)
    phy=sub.add_parser('phy-data',help='Generate 3GPP TDL/CDL statistical channels (proxy labels)')
    phy.add_argument('--output',required=True);phy.add_argument('--seed',type=int,default=42)
    phy.add_argument('--samples-per-profile',type=int,default=80)
    phy.add_argument('--device',default='cpu');phy.add_argument('--spec-version',default='19.2')
    rt=sub.add_parser('rt-data',help='Generate geometric LOS/NLOS from full multi-path RT')
    rt.add_argument('--output',required=True);rt.add_argument('--seed',type=int,default=42)
    rt.add_argument('--scene',default='munich',help='Built-in scene name or scene file')
    rt.add_argument('--tx',type=float,nargs=3,default=[8.5,21.,27.])
    rt.add_argument('--center',type=float,nargs=3,default=[44.,95.,1.5])
    rt.add_argument('--radius',type=float,default=90.)
    rt.add_argument('--per-class',type=int,default=16)
    rt.add_argument('--max-attempts',type=int,default=2000)
    rt.add_argument('--samples-per-src',type=int,default=120000)
    rt.add_argument('--block-size',type=float,default=30.)
    rt.add_argument('--max-depth',type=int,default=3)
    rt.add_argument('--allow-direct-only',action='store_true')
    exp=sub.add_parser('experiment',help='Evaluate MLP vs CORAL MLP and baselines')
    exp.add_argument('--input',nargs='+',required=True,help='One or more dataset NPZs')
    exp.add_argument('--output',required=True)
    exp.add_argument('--holdout',default=None,help='Unseen RT scene/domain, e.g. rt_munich')
    exp.add_argument('--methods',nargs='+',default=['rf','mlp','coral0','coral'])
    exp.add_argument('--seeds',type=int,nargs='+',default=[11,23,47,59,71,83,97,101,109,127])
    exp.add_argument('--bootstrap',type=int,default=200)
    exp.add_argument('--device',default='cpu')
    exp.add_argument('--phase',action='store_true')
    exp.add_argument('--observation',choices=['ideal_cfr','comb_ls'],default='ideal_cfr')
    exp.add_argument('--pilot-spacing',type=int,default=4)
    exp.add_argument('--pilot-snr-db',type=float,default=12.)
    exp.add_argument('--observation-seed',type=int,default=2026)
    args=parser.parse_args()
    if args.command=='doctor':
        from .provenance import versions
        print(json.dumps(versions(),indent=2));return
    if args.command=='city-scene':
        from .geometry import CityConfig,make_city
        p=make_city(args.output_dir, CityConfig(grid_size=args.grid_size,
                     n_vehicles=args.vehicles,seed=args.seed))
        print(f'Controlled synthetic 3D city: {p}');return
    if args.command=='debug-data':
        from .synthetic import debug_multipath
        d=debug_multipath(n_scenes=args.scenes,links_per_scene=args.links_per_scene,seed=args.seed)
        d.save(args.output)
        print(f'DEBUG DATA (not research evidence): saved {len(d.y)} to {args.output}');return
    if args.command=='phy-data':
        from .sionna2 import generate_standardized
        d=generate_standardized(per_profile=args.samples_per_profile,seed=args.seed,
                                 device=args.device,spec_version=args.spec_version)
        d.save(args.output)
        print(f'3GPP PROFILE-PROXY LABELS: saved {len(d.y)} to {args.output}');return
    if args.command=='rt-data':
        from .sionna2 import generate_raytraced, RayScene
        cfg=RayScene(name=args.scene,tx_position=tuple(args.tx),rx_center=tuple(args.center),
                     radius_m=args.radius,per_class=args.per_class,max_attempts=args.max_attempts,
                     samples_per_src=args.samples_per_src,block_size_m=args.block_size,
                     max_depth=args.max_depth,require_los_multipath=not args.allow_direct_only)
        d=generate_raytraced(cfg,seed=args.seed)
        d.save(args.output)
        print(f'GEOMETRY-LABELED RT: saved {len(d.y)} to {args.output}');return
    if args.command=='experiment':
        from .experiment import run_experiment
        result=run_experiment(args.input,args.output,holdout=args.holdout,
                              methods=args.methods,seeds=args.seeds,bootstrap=args.bootstrap,
                              include_phase=args.phase,device=args.device,
                              observation=args.observation,pilot_spacing=args.pilot_spacing,
                              pilot_snr_db=args.pilot_snr_db,
                              observation_seed=args.observation_seed)
        print(json.dumps(result,indent=2));return


if __name__=='__main__':main()
