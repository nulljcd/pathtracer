import javax.swing.JFrame;
import javax.swing.JPanel;

import java.awt.Color;
import java.awt.Dimension;
import java.awt.Graphics;
import java.awt.image.BufferedImage;

public class Viewport extends JFrame {
  private JPanel panel;
  private BufferedImage bufferedImage;
  private Graphics g;

  public Viewport(int width, int height) {
    panel = new JPanel();
    panel.setPreferredSize(new Dimension(width, height));

    setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
    setResizable(false);
    add(panel);
    pack();
    setLocationRelativeTo(null);
    setVisible(true);

    bufferedImage = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);
    g = panel.getGraphics();
  }

  public BufferedImage getBufferedImage() {
    return bufferedImage;
  }

  public void setRGB(int x, int y, int color) {
    bufferedImage.setRGB(x, y, color);
  }

  public void drawComputingRect(int x, int y, int width, int height) {
    Graphics bg = bufferedImage.createGraphics();
    bg.setColor(new Color(200, 200, 255));
    bg.drawRect(x, y, width-1, height-1);
  }

  public void draw() {
    g.drawImage(bufferedImage, 0, 0, null);
  }
}