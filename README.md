# pathtracer
A simple pathtracer.

## specs
- completely vanilla java (yea, it's slow)
- unbiased uni-directional path tracer
- right-handed coord system
- obj support
- bvh support
- simple physically based materials
- tone mapping


## images
![render](https://github.com/nulljcd/pathtracer/assets/169845424/4c0935bc-5722-4794-870f-8d8ad8cfdb96)
![render-1](https://github.com/nulljcd/pathtracer/assets/169845424/f9849ef9-4467-4a19-9540-78ce40d98d92)
### more images to come

## important notes
- the render is saved to the render.png file. note that it will be overwritten on the next render if it's name is not changed and is in the same folder
- the dragon.obj is NOT mine, it is from stanford graphics (https://graphics.stanford.edu/)

## to-do
- TODO: better bvh building (sah)
- TODO: maybe skybox? (it would have to be 256 bit channel though)
