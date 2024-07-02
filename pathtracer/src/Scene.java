import java.util.ArrayList;
import java.util.LinkedList;

public class Scene {
  private Camera camera;
  private ArrayList<Mesh> meshes;
  private AmbientLight ambientLight;

  public Scene() {
    meshes = new ArrayList<>();
  }

  public Camera getCamera() {
    return camera;
  }

  public ArrayList<Mesh> getMeshes() {
    return meshes;
  }

  public AmbientLight getAmbientLight() {
    return ambientLight;
  }

  public void add(Camera camera) {
    this.camera = camera;
  }

  public void add(Mesh mesh) {
    meshes.add(mesh);
  }

  public void add(AmbientLight ambientLight) {
    this.ambientLight = ambientLight;
  }

  public HitInfo intersect(Ray ray) {
    // based on Sebastian Lague's code for traversing a bvh to get an intersection
    // https://www.youtube.com/watch?v=C1H4zIiCOaI
    boolean hitTriangle = false;
    double closestLength = -1;
    Triangle closestTriangle = null;
    Material closestTriangleMeshMaterial = null;

    for (Mesh mesh : meshes) {
      LinkedList<BVHNode> BVHNodeQueue = new LinkedList<>();

      BVHNodeQueue.add(mesh.getRoot());

      double boxLength = BVHNodeQueue.getLast().intersect(ray);
      if (boxLength == -1)
        continue;
      if (closestLength != -1 && boxLength>closestLength)
        continue;

      while (BVHNodeQueue.size() > 0) {
        BVHNode currentNode = BVHNodeQueue.pop();

        if (currentNode.checkIsLeafNode()) {
          for (Triangle triangle : currentNode.getLeaves()) {
            double triangleHitLength = triangle.intersect(ray);
            if (triangleHitLength == -1)
              continue;

            if (!hitTriangle || triangleHitLength < closestLength) {
              hitTriangle = true;
              closestLength = triangleHitLength;
              closestTriangle = triangle;
              closestTriangleMeshMaterial = mesh.getMaterial();
            }
          }
        } else {
          double aLength = currentNode.getChildA().intersect(ray);
          double bLength = currentNode.getChildB().intersect(ray);

          if (aLength == -1 && bLength == -1)
            continue;

          if (closestLength == -1) {
            if (aLength < bLength) {
              BVHNodeQueue.add(currentNode.getChildA());
              BVHNodeQueue.add(currentNode.getChildB());
            } else {
              BVHNodeQueue.add(currentNode.getChildA());
              BVHNodeQueue.add(currentNode.getChildB());
            }
          } else {
            if (aLength < bLength) {
              if (aLength < closestLength)
                BVHNodeQueue.add(currentNode.getChildA());
              if (bLength < closestLength)
                BVHNodeQueue.add(currentNode.getChildB());
            } else {
              if (bLength < closestLength)
                BVHNodeQueue.add(currentNode.getChildB());
              if (aLength < closestLength)
                BVHNodeQueue.add(currentNode.getChildA());
            }
          }

        }
      }
    }

    if (hitTriangle)
      return new HitInfo(closestTriangle.getNormal(), closestLength, closestTriangleMeshMaterial);

    return null;
  }

}
