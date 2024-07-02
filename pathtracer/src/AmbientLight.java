public class AmbientLight {
  private Vector3 color;
  private double strength;

  public AmbientLight(Vector3 color, double strength) {
    this.color = color;
    this.strength = strength;
  }

  public Vector3 getColor() {
    return color;
  }

  public double getStrength() {
    return strength;
  }
}
