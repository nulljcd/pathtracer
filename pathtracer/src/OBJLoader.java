import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;

public class OBJLoader {
  public static ArrayList<Triangle> load(String file) {
    BufferedReader r;
    String line;

    ArrayList<String[]> vertices = new ArrayList<>();
    ArrayList<Triangle> triangles = new ArrayList<>();

    try {
      r = new BufferedReader(new FileReader(file));
      line = r.readLine();
      while (line != null) {
        if (line.contains("v ") && !line.contains("#")) {
          vertices.add(line.split("\\s+"));
        }
        if (line.contains("f ") && !line.contains("#")) {
          String[] fLine = line.split("\\s+");

          String[] vertexAIndex = fLine[1].split("/");
          String[] vertexBIndex = fLine[2].split("/");
          String[] vertexCIndex = fLine[3].split("/");

          String[] vertexA = vertices.get(Integer.parseInt(vertexAIndex[0]) - 1);
          String[] vertexB = vertices.get(Integer.parseInt(vertexBIndex[0]) - 1);
          String[] vertexC = vertices.get(Integer.parseInt(vertexCIndex[0]) - 1);

          Triangle triangle = new Triangle(
              new Vector3(
                  Double.parseDouble(vertexA[1]),
                  Double.parseDouble(vertexA[2]),
                  Double.parseDouble(vertexA[3])),
              new Vector3(
                  Double.parseDouble(vertexB[1]),
                  Double.parseDouble(vertexB[2]),
                  Double.parseDouble(vertexB[3])),
              new Vector3(
                  Double.parseDouble(vertexC[1]),
                  Double.parseDouble(vertexC[2]),
                  Double.parseDouble(vertexC[3])));

          triangles.add(triangle);
        }
        line = r.readLine();
      }
      r.close();
      
      return triangles;
    } catch (IOException e) {
      e.printStackTrace();
    }

    return null;
  }
}