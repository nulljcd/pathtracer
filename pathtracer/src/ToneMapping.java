public class ToneMapping {
  public static Vector3 raw(Vector3 input) {
    Vector3 output = MathA.clamp(input, 0, 1);

    return output;
  }

  // hable tonemapping: https://64.github.io/tonemapping/
  private static Vector3 hableTonemapPartial(Vector3 input) {
    double A = 0.15;
    double B = 0.5;
    double C = 0.1;
    double D = 0.2;
    double E = 0.02;
    double F = 0.3;
    // ((x * (A * x + C * B) + D * E) / (x * (A * x + B) + D * F)) - E / F;
    Vector3 output = (((input.multiply(input.multiply(A).add(C * B)).add(D * E))
        .divide((input.multiply(input.multiply(A).add(B)).add(D * F))))).subtract(E / F);

    return output;
  }

  public static Vector3 hable(Vector3 input) {
    double exposureBias = 10;
    Vector3 toneMapped = hableTonemapPartial(input.multiply(exposureBias));
    double W = 11.2;
    Vector3 whiteScale = Vector3.one().divide(hableTonemapPartial(Vector3.one().multiply(W)));
    Vector3 output = MathA.clamp(toneMapped.multiply(whiteScale), 0, 1);

    return output;
  }

  public static Vector3 simple(Vector3 input) {
    Vector3 output = input.divide(input.add(0.16).multiply(1.02));

    return output;
  }

  public static Vector3 ACESCinematic(Vector3 input) {
    // https://www.shadertoy.com/view/XsGfWV
    Matrix3 m1 = new Matrix3(new double[] {
        0.59719, 0.07600, 0.02840,
        0.35458, 0.90834, 0.13383,
        0.04823, 0.01566, 0.83777
    });

    Matrix3 m2 = new Matrix3(new double[] {
        1.60475, -0.10208, -0.00327,
        -0.53108, 1.10813, -0.07276,
        -0.07367, -0.00605, 1.07602
    });

    Vector3 v = m1.multiply(input);
    Vector3 a = v.multiply(v.add(0.0245786).subtract(0.000090537));
    Vector3 b = v.multiply(v.multiply(0.983729).add(0.4329510)).add(0.238081);

    Vector3 output = MathA.pow(MathA.clamp(m2.multiply(a.multiply(b.reciprocal())), 0, 1),
        new Vector3(1 / 2.2, 1 / 2.2, 1 / 2.2));

    return output;
  }
}
