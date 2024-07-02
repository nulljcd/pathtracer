import java.util.ArrayList;

public class Mesh {
  private BVHNode root;
  private Vector3 position;
  private Vector3 scale;
  private Vector3 rotation;
  private ArrayList<Triangle> triangles;
  private Material material;

  public Mesh(Vector3 position, Vector3 scale, Vector3 rotation, ArrayList<Triangle> triangles, Material material) {
    this.position = position;
    this.scale = scale;
    this.rotation = rotation;
    this.material = material;

    for (int i = 0; i<triangles.size(); i++) {
      triangles.get(i).setA(MathA.rotate(triangles.get(i).getA(), this.rotation));
      triangles.get(i).setB(MathA.rotate(triangles.get(i).getB(), this.rotation));
      triangles.get(i).setC(MathA.rotate(triangles.get(i).getC(), this.rotation));
      
      triangles.get(i).setA(triangles.get(i).getA().multiply(scale).add(position));
      triangles.get(i).setB(triangles.get(i).getB().multiply(scale).add(position));
      triangles.get(i).setC(triangles.get(i).getC().multiply(scale).add(position));

      triangles.get(i).calculateNormal();
    }

    root = new BVHNode();
    root.add(triangles);
    root.buildBVH();
  }

  public BVHNode getRoot() {
    return root;
  }

  public Vector3 getPosition() {
    return position;
  }

  public Vector3 getScale() {
    return scale;
  }

  public Vector3 getRotation() {
    return rotation;
  }

  public ArrayList<Triangle> getTriangles() {
    return triangles;
  }

  public Material getMaterial() {
    return material;
  }
}
