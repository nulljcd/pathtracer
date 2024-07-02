import java.util.ArrayList;

public class BVHNode {
  private Vector3 min;
  private Vector3 max;
  private ArrayList<Triangle> leaves;
  private BVHNode childA;
  private BVHNode childB;

  public BVHNode() {
    leaves = new ArrayList<>();
  }

  public Vector3 getMin() {
    return min;
  }

  public Vector3 getMax() {
    return max;
  }

  public void growToBounds(Triangle triangle) {
    if (min == null)
      min = triangle.getA().copy();
    if (max == null)
      max = triangle.getA().copy();

    min = MathA.min(min, triangle.getA());
    min = MathA.min(min, triangle.getB());
    min = MathA.min(min, triangle.getC());

    max = MathA.max(max, triangle.getA());
    max = MathA.max(max, triangle.getB());
    max = MathA.max(max, triangle.getC());
  }

  public void add(Triangle triangle) {
    leaves.add(triangle);

    growToBounds(triangle);
  }

  public void add(ArrayList<Triangle> triangles) {
    for (Triangle triangle : triangles)
      add(triangle);
  }

  public int getLeavesSize() {
    return leaves.size();
  }

  public ArrayList<Triangle> getLeaves() {
    return leaves;
  }

  public boolean checkIsLeafNode() {
    return leaves != null;
  }

  public double intersect(Ray ray) {
    // https://tavianator.com/2015/ray_box_nan.html
    Vector3 inverseDirection = ray.getInverseDirection();

    double t1 = (min.getX() - ray.getPosition().getX()) * inverseDirection.getX();
    double t2 = (max.getX() - ray.getPosition().getX()) * inverseDirection.getX();
    double tmin = Math.min(t1, t2);
    double tmax = Math.max(t1, t2);
    t1 = (min.getY() - ray.getPosition().getY()) * inverseDirection.getY();
    t2 = (max.getY() - ray.getPosition().getY()) * inverseDirection.getY();
    tmin = Math.max(tmin, Math.min(t1, t2));
    tmax = Math.min(tmax, Math.max(t1, t2));
    t1 = (min.getZ() - ray.getPosition().getZ()) * inverseDirection.getZ();
    t2 = (max.getZ() - ray.getPosition().getZ()) * inverseDirection.getZ();
    tmin = Math.max(tmin, Math.min(t1, t2));
    tmax = Math.min(tmax, Math.max(t1, t2));

    if (tmax >= Math.max(tmin, 0))
      return tmin;

    return -1;
  }

  public BVHNode getChildA() {
    return childA;
  }

  public BVHNode getChildB() {
    return childB;
  }

  public void setChildA(BVHNode childA) {
    this.childA = childA;
  }

  public void setChildB(BVHNode childB) {
    this.childB = childB;
  }

  public void buildBVH() {
    split();
  }

  private void split() {
    // note that this split algorithm is VERY unoptimised!
    double xLength = max.getX() - min.getX();
    double yLength = max.getY() - min.getY();
    double zLength = max.getZ() - min.getZ();
    double maxLen = Math.max(xLength, Math.max(yLength, zLength));
    boolean x = xLength == maxLen;
    boolean y = yLength == maxLen;

    BVHNode newChildA = new BVHNode();
    BVHNode newChildB = new BVHNode();

    if (x) {
      for (Triangle triangle : leaves) {
        double midPoint = (triangle.getA().getX() + triangle.getB().getX() + triangle.getC().getX()) / 3;
        if ((midPoint - min.getX()) < (max.getX() - midPoint))
        newChildA.add(triangle);
        else
        newChildB.add(triangle);
      }
    } else if (y) {
      for (Triangle triangle : leaves) {
        double midPoint = (triangle.getA().getY() + triangle.getB().getY() + triangle.getC().getY()) / 3;
        if ((midPoint - min.getY()) < (max.getY() - midPoint))
        newChildA.add(triangle);
        else
        newChildB.add(triangle);
      }
    } else {
      for (Triangle triangle : leaves) {
        double midPoint = (triangle.getA().getZ() + triangle.getB().getZ() + triangle.getC().getZ()) / 3;
        if ((midPoint - min.getZ()) < (max.getZ() - midPoint))
        newChildA.add(triangle);
        else
        newChildB.add(triangle);
      }
    }

    if (newChildA.getLeavesSize() > 0 && newChildB.getLeavesSize() > 0) {
      this.leaves = null;
      
      childA = newChildA;
      childB = newChildB;

      if (childA.getLeavesSize() > 2)
        childA.split();
      if (childB.getLeavesSize() > 2)
        childB.split();
    }
  }
}
