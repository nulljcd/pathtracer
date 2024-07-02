public class Material {
  private Vector3 color;
  private Vector3 emissionColor;
  private double emissionStrength;
  private Vector3 clearCoatColor;
  private double clearCoatWeight;
  private double clearCoatRoughness;

  // the properties are based on Sebastian Lague's raytracer code for materials
  // https://www.youtube.com/watch?v=Qz0KTGYJtUk
  public Material(Vector3 color, Vector3 clearCoatColor, double clearCoatWeight, double clearCoatRoughness, Vector3 emissionColor, double emissionStrength) {
    this.color = color;
    this.clearCoatColor = clearCoatColor;
    this.clearCoatWeight = clearCoatWeight;
    this.clearCoatRoughness = clearCoatRoughness;
    this.emissionColor = emissionColor;
    this.emissionStrength = emissionStrength;
  }

  public Vector3 getColor() {
    return color;
  }

  public Vector3 getClearCoatColor() {
    return clearCoatColor;
  }

  public double getClearCoatWeight() {
    return clearCoatWeight;
  }

  public double getClearCoatRoughness() {
    return clearCoatRoughness;
  }

  public Vector3 getEmissionColor() {
    return emissionColor;
  }

  public double getEmissionStrength() {
    return emissionStrength;
  }
}
