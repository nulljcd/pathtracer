public class Camera {
  private Vector3 position;
  private Vector3 rotation;
  private int width;
  private int height;
  private double aspect;

  public Camera(Vector3 position, Vector3 rotation, int width, int height) {
    this.position = position;
    this.rotation = rotation;
    this.width = width;
    this.height = height;

    this.aspect = (double) this.width / this.height;
  }

  public Vector3 getPosition() {
    return position;
  }

  public Vector3 getRotation() {
    return rotation;
  }

  public int getWidth() {
    return width;
  }

  public int getHeight() {
    return height;
  }

  public double getAspect() {
    return aspect;
  }
}
