public class MathA {
  public static Vector3 min(Vector3 a, Vector3 b) {
    return new Vector3(
      Math.min(a.getX(),b.getX()), Math.min(a.getY(),b.getY()), Math.min(a.getZ(), b.getZ()));
  }

  public static Vector3 max(Vector3 a, Vector3 b) {
    return new Vector3(
      Math.max(a.getX(),b.getX()), Math.max(a.getY(),b.getY()), Math.max(a.getZ(), b.getZ()));
  }

  public static Vector3 min(Vector3 a, double b) {
    return new Vector3(
      Math.min(a.getX(),b), Math.min(a.getY(),b), Math.min(a.getZ(), b));
  }

  public static Vector3 max(Vector3 a, double b) {
    return new Vector3(
      Math.max(a.getX(),b), Math.max(a.getY(),b), Math.max(a.getZ(), b));
  }

  public static Vector3 clamp(Vector3 a, double min ,double max) {
    return MathA.min(MathA.max(a, min), max);
  }

  public static Vector3 pow(Vector3 a, Vector3 b) {
    return new Vector3(
      Math.pow(a.getX(), b.getX()), 
      Math.pow(a.getY(), b.getY()), 
      Math.pow(a.getZ(), b.getZ()));
  }

  public static Vector3 lerp(Vector3 a, Vector3 b, double f) {
    return a.add((b.subtract(a).multiply(f)));
  }

  public static Matrix3 rotationMatrixX(double angle) {
    return new Matrix3(new double[] {
      1, 0, 0,
      0, Math.cos(angle), -Math.sin(angle),
      0, Math.sin(angle), Math.cos(angle),
    });
  }

  public static Matrix3 rotationMatrixY(double angle) {
    return new Matrix3(new double[] {
      Math.cos(angle), 0, Math.sin(angle),
      0, 1, 0,
      -Math.sin(angle), 0, Math.cos(angle)
    });
  }

  public static Matrix3 rotationMatrixZ(double angle) {
    return new Matrix3(new double[] {
      Math.cos(angle), -Math.sin(angle), 0,
      Math.sin(angle), Math.cos(angle), 0,
      0, 0, 1
    });
  }

  public static Vector3 rotate(Vector3 point, Vector3 rotation) {
    Matrix3 a = MathA.rotationMatrixX(rotation.getX());
    Matrix3 b = MathA.rotationMatrixY(rotation.getY());
    Matrix3 c = MathA.rotationMatrixZ(rotation.getZ());
    
    return a.multiply(b).multiply(c).multiply(point);
  }
}
