public class Matrix3 {
  private double[] values;

  public Matrix3(double[] values) {
    this.values = values;
  }

  public static Matrix3 zero() {
    return new Matrix3(new double[] {0,0,0, 0,0,0, 0,0,0});
  } 

  public Matrix3 multiply(Matrix3 other) {
    // http://blog.rogach.org/2015/08/how-to-create-your-own-simple-3d-render.html
    double[] result = new double[9];
    for (int row = 0; row < 3; row++) {
        for (int col = 0; col < 3; col++) {
            for (int i = 0; i < 3; i++) {
                result[row * 3 + col] += this.values[row * 3 + i] * other.values[i * 3 + col];
            }
        }
    }
    return new Matrix3(result);
  }

  public Vector3 multiply(Vector3 other) {
    return new Vector3(
        other.getX() * values[0] + other.getY() * values[3] + other.getZ() * values[6],
        other.getX() * values[1] + other.getY() * values[4] + other.getZ() * values[7],
        other.getX() * values[2] + other.getY() * values[5] + other.getZ() * values[8]);
  }
}
