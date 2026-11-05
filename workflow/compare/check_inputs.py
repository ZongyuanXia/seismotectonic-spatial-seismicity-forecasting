"""Check A/B/C transformations on real stored frames; no model execution."""
import argparse
import sys
from pathlib import Path
from compare_common import HERE, DEFAULT_PROJECT, frozen


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, default=DEFAULT_PROJECT / 'cl10_mapping_v2')
    p.add_argument('--all-frames', action='store_true', help='Scan every unique input frame')
    a = p.parse_args()
    frozen()
    sys.path.insert(0, str(HERE / 'vendor'))
    import numpy as np
    from compare_data import placeholder_values
    paths = sorted((a.data / 'frames').glob('*.npz'))
    if not paths:
        raise ValueError('No input frames')
    if not a.all_frames:
        paths = [paths[i] for i in sorted({0, len(paths)//4, len(paths)//2, 3*len(paths)//4, len(paths)-1})]
    # Includes different valid masks and the main direction pair.
    selections = [[0, i] for i in range(1, 10)] + [[0, 1, 2], [0, 3, 9], [0, 4, 8]]
    for path in paths:
        with np.load(path, allow_pickle=False) as frame:
            gray, valid = frame['gray'], frame['valid']
        for channels in selections:
            for group, constant in [('A', 0), ('B', 100), ('C', 255)]:
                result = placeholder_values(gray, valid, channels, group)
                assert np.array_equal(result[:, 0], gray[:, 0]), 'MAG changed'
                expected = np.zeros_like(result[:, 1:]) if group == 'A' else np.where(
                    valid[:, np.asarray(channels[1:])-1], constant, 0).astype(np.float32)
                assert np.array_equal(result[:, 1:], expected)
                assert np.isfinite(result).all() and np.all((result >= 0) & (result <= 255))
    print(f'Checked {len(paths)} frames, {len(selections)} channel selections, all three groups. '
          'MAG and support rules passed. No training, inference or scoring.')


if __name__ == '__main__':
    main()
