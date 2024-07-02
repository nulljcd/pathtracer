import java.io.File;
import java.io.IOException;
import java.util.Random;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

import javax.imageio.ImageIO;

public class Renderer {
  public static void render(Viewport viewport, Scene scene, int raysPerPixel, int maxBounces, int toneMappingType, int blockSize,
      int numThreads) {

    try (
        ExecutorService threadPool = Executors.newFixedThreadPool(numThreads)) {
      for (int y = 0; y < scene.getCamera().getHeight(); y += blockSize) {
        for (int x = 0; x < scene.getCamera().getWidth(); x += blockSize) {
          int blockX = x;
          int blockY = y;
          int blockWidth = scene.getCamera().getWidth() < blockX + blockSize ? scene.getCamera().getWidth() - blockX : blockSize;
          int blockHeight = scene.getCamera().getHeight() < blockY + blockSize ? scene.getCamera().getHeight() - blockY : blockSize;

          threadPool.execute(new Runnable() {
            public void run() {
              viewport.drawComputingRect(blockX, blockY, blockWidth, blockHeight);
              viewport.draw();

              for (int y = blockY; y < blockY + blockHeight; y++) {
                for (int x = blockX; x < blockX + blockWidth; x++) {
                  Vector3 uv = new Vector3((double) x / (scene.getCamera().getWidth() - 1), (double) y / (scene.getCamera().getHeight() - 1), 0).multiply(new Vector3(2, -2, 0)).add(new Vector3(-1, 1, 0));
                 
                  Vector3 position = scene.getCamera().getPosition();
                  Vector3 direction = new Vector3(uv.getX() * scene.getCamera().getAspect(), uv.getY(), -2.8).normal(); // note that this contains an arbitrary number for the fov that "looks right"
                  direction = MathA.rotate(direction, scene.getCamera().getRotation());
                  Ray ray = new Ray(position, direction);

                  Vector3 color = Vector3.zero();
                  for (int i = 0; i < raysPerPixel; i++){
                    color = color.add(traceRay(scene, ray, maxBounces));
                  }
                  color = color.multiply((double) 1 / raysPerPixel);
                  
                  switch (toneMappingType) {
                    case 0:
                      color = ToneMapping.raw(color);
                      break;
                    case 1:
                      color = ToneMapping.simple(color);
                      break;
                    case 2:
                      color = ToneMapping.hable(color);
                      break;
                    case 3:
                      color = ToneMapping.ACESCinematic(color);
                      break;
                    default:
                      color = ToneMapping.raw(color);
                      break;
                  }

                  viewport.setRGB(x, y, color.getInt());
                }
              }

              viewport.draw();
            }
          });
        }
      }

      threadPool.shutdown();
    } finally {
      viewport.draw();
      File outputFile = new File("render.png");
      try {
        ImageIO.write(viewport.getBufferedImage(), "png", outputFile);
      } catch (IOException e) {
        e.printStackTrace();
      }
    }
  }

  private static Vector3 traceRay(Scene scene, Ray ray, int maxBounces) {
    Random random = new Random();

    Vector3 incomingLight = Vector3.zero();
    Vector3 rayColor = Vector3.one();

    for (int bounce = 0; bounce < maxBounces; bounce++) {
      HitInfo hitInfo = scene.intersect(ray);

      if (hitInfo != null) {
        // this bit is based on Sebastian Lague's raytracer code for calculating the color of the pixel
        // https://www.youtube.com/watch?v=Qz0KTGYJtUk
        int isSpecularBounce = hitInfo.getMaterial().getClearCoatWeight() >= random.nextDouble() ? 1 : 0;
        Vector3 diffuseDirection = hitInfo.getNormal().add(randomVector3NormalDistribution()).normal();
        Vector3 specularDirection = ray.getDirection().subtract(hitInfo.getNormal().multiply(2 * ray.getDirection().dot(hitInfo.getNormal())));
        Vector3 emittedLight = hitInfo.getMaterial().getEmissionColor().multiply(hitInfo.getMaterial().getEmissionStrength());
        incomingLight = incomingLight.add(emittedLight.multiply(rayColor));
        rayColor = rayColor.multiply(MathA.lerp(hitInfo.getMaterial().getColor(), hitInfo.getMaterial().getClearCoatColor(), isSpecularBounce));

        Vector3 newPosition = ray.getPosition().add(ray.getDirection().multiply(hitInfo.getLength()));
        Vector3 newDirection = MathA.lerp(diffuseDirection, specularDirection, (1 - hitInfo.getMaterial().getClearCoatRoughness()) * isSpecularBounce);
        ray = new Ray(newPosition, newDirection);
      } else {
        Vector3 emittedLight = scene.getAmbientLight().getColor().multiply(scene.getAmbientLight().getStrength());
        incomingLight = incomingLight.add(emittedLight.multiply(rayColor));
        break;
      }
    }

    return incomingLight;
  }

  private static Vector3 randomVector3NormalDistribution() {
    Random random = new Random();

    return new Vector3(
      random.nextGaussian(), 
      random.nextGaussian(), 
      random.nextGaussian()).normal();
  }
}