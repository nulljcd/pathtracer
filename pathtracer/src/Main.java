class Main {
  public static void main(String[] args) {
    // final int viewportWidth = 1920;
    // final int viewportHeight = 1080;
    final int viewportWidth = 1024;
    final int viewportHeight = 640;

    Viewport viewport = new Viewport(viewportWidth, viewportHeight);

    Scene scene = new Scene();
    
    scene.add(new AmbientLight(new Vector3(.3, .4, .5), .4));

    scene.add(new Camera(
			new Vector3(0, 0, 0),
			new Vector3(0, 0, 0),
			viewportWidth, viewportHeight));

    Mesh mesh0 = new Mesh(
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

    scene.add(mesh0);

    Mesh mesh1 = new Mesh(
      new Vector3(0, 0, -6),
      new Vector3(1, 1, 1),
      new Vector3(0, 0, 0),
      OBJLoader.load("res/sphere.obj"),
      new Material(
        new Vector3(1, .5, .7),
        new Vector3(1, 1, 1),
        .2, 
        .3,
        new Vector3(0, 0, 0),
        0));

    scene.add(mesh1);

    Mesh mesh2 = new Mesh(
      new Vector3(-10, 6, -6),
      new Vector3(3, 3, 3),
      new Vector3(0, 0, 0),
      OBJLoader.load("res/sphere.obj"),
      new Material(
        new Vector3(0, 0, 0),
        new Vector3(0, 0, 0),
        0, 
        0,
        new Vector3(1, .6, .2),
        4));

    scene.add(mesh2);

    Renderer.render(viewport, scene, 200, 10,1, 64, 4);
  }
}
