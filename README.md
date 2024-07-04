# pathtracer
A simple pathtracer in vanilla java.

## specs
- completely vanilla java (yea, it's slow)
- multithreading
- unbiased uni-directional path tracing
- right-handed coord system
- obj support
- bvh building
- simple physically based materials
- tone mapping
- easy to customize

## images
![render](https://github.com/nulljcd/pathtracer/assets/169845424/4c0935bc-5722-4794-870f-8d8ad8cfdb96)
![render-1](https://github.com/nulljcd/pathtracer/assets/169845424/f9849ef9-4467-4a19-9540-78ce40d98d92)
dragon from (https://casual-effects.com)
![render](https://github.com/nulljcd/pathtracer/assets/169845424/a70d49c6-5daf-471d-ba67-f86ce61ac722)
bmw from (https://casual-effects.com)

### Tone Mapping Comparison
Raw output
![render-1](https://github.com/nulljcd/pathtracer/assets/169845424/69f5ae22-b1fc-4e29-b14e-174e9dc46400)
Simple tonemapping
![render](https://github.com/nulljcd/pathtracer/assets/169845424/28be0a95-337e-4745-9221-ca6b55d6f893)
Hable tonemapping
![render](https://github.com/nulljcd/pathtracer/assets/169845424/455d4875-dccf-4233-843b-f12ade60d9a1)
ACES cinematic tonemapping
![aces](https://github.com/nulljcd/pathtracer/assets/169845424/9ae6d68f-eaf0-4879-bc28-edcb24c7764d)

## important notes
- the render is saved to the render.png file. note that it will be overwritten on the next render if it's name is not changed and is in the same folder

## to-do
- TODO: better bvh building (sah)
- TODO: maybe skybox? (it would have to be 256 bit channel though)
- TODO: antialiasing
