public class Triangle {
  private Vector3 a;
  private Vector3 b;
  private Vector3 c;
  private Vector3 normal;

  public Triangle(Vector3 a, Vector3 b, Vector3 c) {
    this.a = a;
    this.b = b;
    this.c = c;
  }

  public Vector3 getA() {
    return a;
  }

  public Vector3 getB() {
    return b;
  }

  public Vector3 getC() {
    return c;
  }
  
  public void setA(Vector3 a) {
    this.a = a;
  }

  public void setB(Vector3 b) {
    this.b = b;
  }

  public void setC(Vector3 c) {
    this.c = c;
  }

  public void calculateNormal() {
    normal = b.subtract(a).cross(c.subtract(a)).normal();
  }

  public Vector3 getNormal() {
    return normal;
  }

  public double intersect(Ray ray) {
    // https://en.wikipedia.org/wiki/M%C3%B6ller%E2%80%93Trumbore_intersection_algorithm
    double epsilon = 0.0000001;
    Vector3 edge1 = this.b.subtract(this.a);
    Vector3 edge2 = this.c.subtract(this.a);
    Vector3 h = ray.getDirection().cross(edge2);
    double a = edge1.dot(h);

    if (a > -epsilon && a < epsilon)
     return -1;
    
     double f = 1 / a;
    Vector3 s = ray.getPosition().subtract(this.a);
    double u = f * (s.dot(h));

    if (u < 0 || 1 < u)
     return -1;

    Vector3 q = s.cross(edge1);
    double v = f * ray.getDirection().dot(q);

    if (v < 0 || u + v > 1)
     return -1;

    double t = f * edge2.dot(q);
    if (t > epsilon)
      return t;
      
   return -1;
  }
}
