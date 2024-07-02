public class Vector3 {
  private double x;
  private double y;
  private double z;

  public Vector3(double x, double y, double z) {
    this.x = x;
    this.y = y;
    this.z = z;
  }

  public double getX() {
    return x;
  }

  public double getY() {
    return y;
  }

  public double getZ() {
    return z;
  }

  public void setX(double x) {
    this.x = x;
  }

  public void setY(double y) {
    this.y = y;
  }

  public void setZ(double z) {
    this.z = z;
  }

  public Vector3 add(Vector3 other) {
    return new Vector3(x + other.x, y + other.y, z + other.z);
  }

  public Vector3 subtract(Vector3 other) {
    return new Vector3(x - other.x, y - other.y, z - other.z);
  }

  public Vector3 multiply(Vector3 other) {
    return new Vector3(x * other.x, y * other.y, z * other.z);
  }

  public Vector3 divide(Vector3 other) {
    return this.multiply(other.reciprocal());
  }

  public Vector3 add(double other) {
    return new Vector3(x + other, y + other, z + other);
  }

  public Vector3 subtract(double other) {
    return new Vector3(x - other, y - other, z - other);
  }

  public Vector3 multiply(double other) {
    return new Vector3(x * other, y * other, z * other);
  }

  public Vector3 divide(double other) {
    return this.multiply(1/other);
  }

  public double length2() {
    return this.dot(this);
  }

  public double dot(Vector3 other) {
    return other.x * this.x + other.y * this.y + other.z * this.z;
  }

  public Vector3 cross(Vector3 other) {
    return new Vector3(this.y * other.z - this.z * other.y, this.z * other.x
        - this.x * other.z, this.x * other.y - this.y * other.x);
  }

  public Vector3 normal() {
    double invLength = 1 / Math.sqrt(this.length2());
    return new Vector3(x * invLength, y * invLength, z * invLength);
  }

  public Vector3 reverse() {
    return new Vector3(-x, -y, -z);
  }
  
  public Vector3 reciprocal() {
    return new Vector3(1/x, 1/y, 1/z);
  }

  public Vector3 copy() {
    return new Vector3(x, y, z);
  }

  public int getInt() {
    return ((((int) (255)) & 0xFF) << 24) |
        ((((int) (x * 255)) & 0xFF) << 16) |
        ((((int) (y * 255)) & 0xFF) << 8) |
        ((((int) (z * 255)) & 0xFF) << 0);
  }

  public static Vector3 zero() {
    return new Vector3(0, 0, 0);
  }

  public static Vector3 one() {
    return new Vector3(1, 1, 1);
  }
}