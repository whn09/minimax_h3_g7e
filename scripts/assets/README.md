# Sample condition materials

Fixtures for the fl2va / ref2va examples, so the commands in the guides run without hunting for
inputs. All of them derive from one generated 864x480 t2va clip (`mkmat.sh` cut the frames and the
audio out of `ref.mp4`), so they are throwaway test material, not reference content.

| file | what it is | use |
|---|---|---|
| `first.png` | 864x480, frame 0 of `ref.mp4` | `--task fl2va --image` |
| `last.png` | 864x480, final frame | `--task fl2va --last-image` |
| `ref.mp4` | 864x480, 243 frames, 10.125 s, with AAC audio | `--task ref2va --ref-video` |
| `ref5s.mp4` | the same clip truncated to 5.04 s | ref2va at the short clip length — output length follows the reference |
| `refaudio.wav` | 16 kHz mono PCM, extracted from `ref.mp4` | `--task ref2va --ref-audio` (duration is then derived from it) |

Send them with `h3gen.py --inline`, which puts the bytes in the request body as a `data:` URI. A bare
path is resolved **on the server**, so it only works when the server can see that path. See §0.3 of
`DEPLOYMENT_GUIDE.md` for all the URI schemes the server accepts.
