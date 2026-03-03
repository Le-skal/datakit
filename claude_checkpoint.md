# Claude Checkpoint — OOP for AI (cours)

## Project
Root: `C:/Users/rmartin/Desktop/code agefi/cours/`
Goal: OOP final project — dataset hierarchy, BatchLoader, preprocessing pipeline.

---

## Datasets available
| Path | Class | Mode | Labels |
|---|---|---|---|
| `dataset/Oxford-IIIT-Pet/` | `ImageDataset` | CSV | `dataset/oxford_labels.csv` (breed string) |
| `dataset/UTKFace/` | `ImageDataset` | CSV | need to generate (age int from filename) |
| `dataset/ESC-50/` | `AudioDataset` | CSV | `ESC-50/meta/esc50.csv` (category string) |
| `dataset/BallroomData/` | `AudioDataset` | folder hierarchy | genre string |
| `dataset/BallroomData/` | `AudioDataset` | CSV | BPM float — compute from `BallroomAnnotations/` .beats files |

---

## Implementation Status

### Done ✅
- `src/utils.py` — `check_type`, `check_range`, `parse_labels_csv`, `load_image`
- `src/dataset.py` — `Dataset(ABC)`, `LabeledDataset(ABC)`, `UnlabeledDataset(ABC)`
- `src/image_dataset.py` — `ImageDataset`, `UnlabeledImageDataset`
- `src/audio_dataset.py` — `AudioDataset`, `UnlabeledAudioDataset`
- `src/batch_loader.py` — `BatchLoader` (shuffle, drop_last, `__iter__`, `__len__`)
- `src/preprocessing.py` — `Transform(ABC)`, `CenterCrop`, `RandomCrop`, `RandomFlip`, `Padding`, `MelSpectrogram`, `AudioRandomCrop`, `Resample`, `PitchShift`, `Pipeline`
- `main.py` — full showcase: CSV generation, datasets, BatchLoader, pipelines
- All README checkboxes ticked

### TODO
- `report.md` — design decisions, workload split, usage documentation
- Sphinx HTML docs — run on a good PC for +0.5 bonus points

---

## Todo list

- [ ] **Sphinx** — generate HTML docs from existing docstrings (+0.5 bonus). Do on a good PC.
- [x] **report.md** — written with full design decisions, results, images, usage examples.
- [x] **Generic indications checkboxes** — encapsulation, type checks, docstrings, report all ticked.
- [ ] **Commit & push** after each of the above.

---

## Key Design Decisions
- `split()` shuffles indices with `random.shuffle`, then slices; `_create_subset` uses `object.__new__` to bypass `__init__`
- CSV labels: auto-cast int → float → str at load time
- Audio `_load_file` returns `(np.ndarray, int)` — librosa-compatible `(y, sr)` tuple
- `LabeledDataset.__init__` initialises `self._labels = []` before `super().__init__()` so `_scan_files` runs first; `_load_labels()` is called after

---

## Dependencies
`pillow`, `numpy`, `librosa` — all installed.
