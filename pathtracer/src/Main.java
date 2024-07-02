class Main {
  public static void main(String[] args) {
  // --- specs ---
  // completely vanilla java (yea, it's slow)
  // unbiased uni-directional path tracer
  // right-handed coord system
  // obj support
  // bvh support
  // simple physically based materials
  // tone mapping

  // --- important notes ---
  // - the render is saved to the render.png file. note that it will be overwritten on the next render if it's name is not changed and is in the same folder
  // - the dragon.obj is NOT mine, it is from stanford graphics (https://graphics.stanford.edu/)


  // --- to-do ---
  // TODO: better bvh building (sah)
  // TODO: maybe skybox? (it would have to be 256 bit channel though)
  // TODO: webgl version?

    final int viewportWidth = 1024;
    final int viewportHeight = 640;

    Viewport viewport = new Viewport(viewportWidth, viewportHeight);

    Scene scene = new Scene();
    // the scene is setup with the render from the readme.md in github

    scene.add(new AmbientLight(new Vector3(.2, .3, .4), 1));

    scene.add(new Camera(
			new Vector3(0, 0, 0),
			new Vector3(0, 0, 0),
			viewportWidth, viewportHeight));

    Mesh mesh0 = new Mesh(
      new Vector3(0, 0, -6), // position
      new Vector3(2,2,2), // scale
      new Vector3(0, -.5, 0), // rotation
      OBJLoader.load("res/dragon.obj"), // triangles
      new Material(
        new Vector3(0, 0, 0), // color
        new Vector3(.8, .8, .8), // clear coat color
        1, // clear coat weight
        .4, // clear coat roughness
        new Vector3(0, 0, 0), // emission color
        0)); // emission strength

    scene.add(mesh0);

    Mesh mesh1 = new Mesh(
      new Vector3(-10, 6, -6),
      new Vector3(3, 3, 3),
      new Vector3(0, 0, 0),
      OBJLoader.load("res/sphere.obj"),
      new Material(
        new Vector3(0, 0, 0),
        new Vector3(0, 0, 0),
        0, 
        0,
        new Vector3(1, .6, .1),
        4));

    scene.add(mesh1);

    Mesh mesh2 = new Mesh(
      new Vector3(0, -1, -6),
      new Vector3(10, 1, 10),
      new Vector3(0, 0, 0),
      OBJLoader.load("res/plane.obj"),
      new Material(
        new Vector3(1, 1, 1),
        new Vector3(0, 0, 0),
        0, 
        0,
        new Vector3(0, 0, 0),
        0));

    scene.add(mesh2);

    // --- options for rendering ---
    // viewport,
    // scene,
    // rays per pixel,
    // max bounces for a ray
    // tone mapping type
    // render block size
    // number of threads
    Renderer.render(viewport, scene, 50, 6,1, 64, 4);
  }
}