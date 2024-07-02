public class Ray {
  private Vector3 position;
  private Vector3 direction;
  private Vector3 inverseDirection;

  public Ray(Vector3 position, Vector3 direction) {
    this.position = position;
    this.direction = direction;
    inverseDirection = direction.reciprocal();
  }
  
  public Vector3 getPosition() {
    return position;
  }

  public Vector3 getDirection() {
    return direction;
  }

  public Vector3 getInverseDirection() {
    return inverseDirection;
  }
}
