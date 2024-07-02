public class HitInfo {
  private Vector3 normal;
  private double length;
  private Material material;

  public HitInfo(Vector3 normal, double length, Material material) {
    this.normal = normal;
    this.length = length;
    this.material = material;
  }

  public Vector3 getNormal() {
    return normal;
  }

  public double getLength() {
    return length;
  }

  public Material getMaterial() {
    return material;
  }
}
