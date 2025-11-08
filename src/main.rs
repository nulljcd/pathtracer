use glam::{Mat3, Vec3};
use image::{self, DynamicImage, ExtendedColorType, ImageError, ImageReader, save_buffer};
use indicatif::ProgressBar;
use nanorand::{Rng, WyRand};
use rayon::{
    iter::{IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator},
    slice::ParallelSliceMut,
};
use std::{f32, fs, time::Instant};

#[derive(Clone, Copy)]
pub struct Ray {
    pub origin: Vec3,
    pub direction: Vec3,
    pub inverse_direction: Vec3,
}

impl Ray {
    pub fn new(origin: Vec3, direction: Vec3) -> Self {
        Ray {
            origin,
            direction,
            inverse_direction: Vec3::ZERO,
        }
    }

    pub fn update_internal_data(&mut self) {
        self.inverse_direction = Vec3::ONE / self.direction;
    }
}

pub struct BoundingBox {
    pub min: Vec3,
    pub max: Vec3,
}

impl BoundingBox {
    pub fn new(min: Vec3, max: Vec3) -> Self {
        BoundingBox { min, max }
    }

    pub fn empty() -> Self {
        BoundingBox {
            min: Vec3::INFINITY,
            max: Vec3::NEG_INFINITY,
        }
    }

    pub fn intersect_ray(&self, ray: &Ray, nearest_hit_distance: f32) -> Option<f32> {
        let tx0: f32 = (self.min.x - ray.origin.x) * ray.inverse_direction.x;
        let tx1: f32 = (self.max.x - ray.origin.x) * ray.inverse_direction.x;
        let ty0: f32 = (self.min.y - ray.origin.y) * ray.inverse_direction.y;
        let ty1: f32 = (self.max.y - ray.origin.y) * ray.inverse_direction.y;
        let tz0: f32 = (self.min.z - ray.origin.z) * ray.inverse_direction.z;
        let tz1: f32 = (self.max.z - ray.origin.z) * ray.inverse_direction.z;
        let tmin: f32 = tx0.min(tx1).max(ty0.min(ty1)).max(tz0.min(tz1)).max(0.);
        (tx0.max(tx1).min(ty0.max(ty1)).min(tz0.max(tz1)) >= tmin && tmin < nearest_hit_distance)
            .then_some(tmin)
    }
}

pub struct Triangle {
    pub vertex1: Vec3,
    pub vertex2: Vec3,
    pub vertex3: Vec3,
    pub u: Vec3,
    pub v: Vec3,
    pub uu: f32,
    pub uv: f32,
    pub vv: f32,
    pub normal: Vec3,
    pub d1: f32,
    pub inv_d2: f32,
    pub min: Vec3,
    pub max: Vec3,
    pub centroid: Vec3,
}

impl Triangle {
    pub fn new(vertex1: Vec3, vertex2: Vec3, vertex3: Vec3) -> Self {
        let u: Vec3 = vertex2 - vertex1;
        let v: Vec3 = vertex3 - vertex1;
        let uu: f32 = u.length_squared();
        let uv: f32 = u.dot(v);
        let vv: f32 = v.length_squared();
        let normal: Vec3 = u.cross(v).normalize();
        let denom: f32 = uu * vv - uv * uv;

        Triangle {
            vertex1,
            vertex2,
            vertex3,
            u,
            v,
            uu,
            uv,
            vv,
            normal,
            d1: -vertex1.dot(normal),
            inv_d2: 1. / denom,
            min: vertex1.min(vertex2).min(vertex3),
            max: vertex1.max(vertex2).max(vertex3),
            centroid: (vertex1 + vertex2 + vertex3) / 3.,
        }
    }

    pub fn intersect_ray(&self, ray: &Ray, nearest_hit_distance: f32) -> Option<(f32, Vec3)> {
        let den: f32 = self.normal.dot(ray.direction);
        if den >= 0. {
            return None;
        }
        let distance: f32 = -(self.normal.dot(ray.origin) + self.d1) / den;
        if distance < 0. || distance > nearest_hit_distance {
            return None;
        }
        let w: Vec3 = ray.origin + ray.direction * distance - self.vertex1;
        let wu: f32 = w.dot(self.u);
        let wv: f32 = w.dot(self.v);
        let s: f32 = (self.vv * wu - self.uv * wv) * self.inv_d2;
        let t: f32 = (self.uu * wv - self.uv * wu) * self.inv_d2;
        if s < 0. || t < 0. || (s + t) > 1. {
            return None;
        }
        Some((distance, self.normal))
    }
}

pub struct Node {
    pub left: usize,
    pub right: usize,
    pub is_leaf: bool,
}

pub struct Geometry {
    pub triangles: Vec<Triangle>,
    pub nodes: Vec<Node>,
    pub bounding_boxes: Vec<BoundingBox>,
}

impl Geometry {
    pub fn new(triangles: &Vec<Triangle>, position: Vec3, scale: Vec3, rotation: Mat3) -> Self {
        let triangles: Vec<Triangle> = triangles
            .par_iter()
            .map(|triangle| {
                Triangle::new(
                    rotation * triangle.vertex1 * scale + position,
                    rotation * triangle.vertex2 * scale + position,
                    rotation * triangle.vertex3 * scale + position,
                )
            })
            .collect();

        let num_nodes: usize = 2 * triangles.len() - 1;
        let mut nodes: Vec<Node> = Vec::with_capacity(num_nodes);
        let mut bounding_boxes: Vec<BoundingBox> = Vec::with_capacity(num_nodes);

        let mut triangle_indices_stack: Vec<Vec<usize>> = vec![(0..triangles.len()).collect()];

        while let Some(triangle_indices) = triangle_indices_stack.pop() {
            let triangle_indices_len: usize = triangle_indices.len();

            if triangle_indices_len == 1 {
                let triangle_index: usize = triangle_indices[0];
                let triangle: &Triangle = &triangles[triangle_index];

                let bounding_box: BoundingBox = BoundingBox::new(triangle.min, triangle.max);

                nodes.push(Node {
                    left: triangle_index,
                    right: 0,
                    is_leaf: true,
                });
                bounding_boxes.push(bounding_box);
            } else {
                let mut bounding_box: BoundingBox = BoundingBox::empty();
                let mut triangle_centers_bounding_box: BoundingBox = BoundingBox::empty();

                for triangle_index in &triangle_indices {
                    let triangle: &Triangle = &triangles[*triangle_index];

                    bounding_box.min = bounding_box.min.min(triangle.min);
                    bounding_box.max = bounding_box.max.max(triangle.max);
                    triangle_centers_bounding_box.min =
                        triangle_centers_bounding_box.min.min(triangle.centroid);
                    triangle_centers_bounding_box.max =
                        triangle_centers_bounding_box.max.max(triangle.centroid);
                }

                let bounding_box_size: Vec3 = bounding_box.max - bounding_box.min;
                let bounding_box_area: f32 = 2.
                    * (bounding_box_size.x * bounding_box_size.y
                        + bounding_box_size.x * bounding_box_size.z
                        + bounding_box_size.y * bounding_box_size.z);

                let triangle_centers_bounding_box_size: Vec3 =
                    triangle_centers_bounding_box.max - triangle_centers_bounding_box.min;
                let triangle_centers_bounding_box_center: Vec3 =
                    (triangle_centers_bounding_box.min + triangle_centers_bounding_box.max) * 0.5;

                let mut lowest_cost: f32 = f32::MAX;
                let mut split_data: Option<(Vec<usize>, Vec<usize>)> = None;

                for split_axis in 0..3 {
                    for split_test in 0..7 {
                        let offset: f32 = split_test as f32 / (7 / 2) as f32 - 1.;
                        let split_position: f32 = triangle_centers_bounding_box_center[split_axis]
                            + offset * triangle_centers_bounding_box_size[split_axis] * 0.75;

                        let mut left_triangle_indices: Vec<usize> = Vec::new();
                        let mut right_triangle_indices: Vec<usize> = Vec::new();

                        for triangle_index in &triangle_indices {
                            let triangle: &Triangle = &triangles[*triangle_index];
                            if triangle.centroid[split_axis] < split_position {
                                left_triangle_indices.push(*triangle_index);
                            } else {
                                right_triangle_indices.push(*triangle_index);
                            }
                        }

                        if left_triangle_indices.is_empty() {
                            for _ in 0..right_triangle_indices.len() / 2 {
                                left_triangle_indices.push(right_triangle_indices.pop().unwrap());
                            }
                        } else if right_triangle_indices.is_empty() {
                            for _ in 0..left_triangle_indices.len() / 2 {
                                right_triangle_indices.push(left_triangle_indices.pop().unwrap());
                            }
                        }

                        let mut left_bounding_box: BoundingBox = BoundingBox::empty();
                        let mut right_bounding_box: BoundingBox = BoundingBox::empty();

                        for triangle_index in &left_triangle_indices {
                            let triangle: &Triangle = &triangles[*triangle_index];
                            left_bounding_box.min = left_bounding_box.min.min(triangle.min);
                            left_bounding_box.max = left_bounding_box.max.max(triangle.max);
                        }

                        for triangle_index in &right_triangle_indices {
                            let triangle: &Triangle = &triangles[*triangle_index];
                            right_bounding_box.min = right_bounding_box.min.min(triangle.min);
                            right_bounding_box.max = right_bounding_box.max.max(triangle.max);
                        }

                        let left_bounding_box_size: Vec3 =
                            left_bounding_box.max - left_bounding_box.min;
                        let right_bounding_box_size: Vec3 =
                            right_bounding_box.max - right_bounding_box.min;
                        let left_bounding_box_area: f32 = 2.
                            * (left_bounding_box_size.x * left_bounding_box_size.y
                                + left_bounding_box_size.x * left_bounding_box_size.z
                                + left_bounding_box_size.y * left_bounding_box_size.z);
                        let right_bounding_box_area: f32 = 2.
                            * (right_bounding_box_size.x * right_bounding_box_size.y
                                + right_bounding_box_size.x * right_bounding_box_size.z
                                + right_bounding_box_size.y * right_bounding_box_size.z);
                        let cost: f32 = (left_bounding_box_area / bounding_box_area)
                            * left_triangle_indices.len() as f32
                            + (right_bounding_box_area / bounding_box_area)
                                * right_triangle_indices.len() as f32;

                        if cost < lowest_cost {
                            lowest_cost = cost;
                            split_data = Some((left_triangle_indices, right_triangle_indices));
                        }
                    }
                }

                let (left_triangle_indices, right_triangle_indices) = split_data.unwrap();

                nodes.push(Node {
                    left: nodes.len() + 1,
                    right: nodes.len() + 2 * left_triangle_indices.len(),
                    is_leaf: false,
                });
                bounding_boxes.push(bounding_box);
                triangle_indices_stack.push(right_triangle_indices);
                triangle_indices_stack.push(left_triangle_indices);
            }
        }

        Geometry {
            triangles,
            nodes,
            bounding_boxes,
        }
    }

    pub fn intersect_ray(&self, ray: &Ray, mut nearest_hit_distance: f32) -> Option<(f32, Vec3)> {
        self.bounding_boxes[0]
            .intersect_ray(ray, nearest_hit_distance)?;

        let mut nearest_hit_data: Option<(f32, Vec3)> = None;
        let mut stack: Vec<usize> = Vec::with_capacity(128);

        stack.push(0);

        while let Some(node_index) = stack.pop() {
            let node: &Node = &self.nodes[node_index];

            if node.is_leaf {
                if let Some((hit_distance, hit_normal)) =
                    self.triangles[node.left].intersect_ray(ray, nearest_hit_distance)
                {
                    nearest_hit_distance = hit_distance;
                    nearest_hit_data = Some((hit_distance, hit_normal));
                }
            } else {
                let left_node_index: usize = node.left;
                let right_node_index: usize = node.right;

                match (
                    self.bounding_boxes[left_node_index].intersect_ray(ray, nearest_hit_distance),
                    self.bounding_boxes[right_node_index].intersect_ray(ray, nearest_hit_distance),
                ) {
                    (Some(left_hit_distance), Some(right_hit_distance)) => {
                        if left_hit_distance < right_hit_distance {
                            stack.push(right_node_index);
                            stack.push(left_node_index);
                        } else {
                            stack.push(left_node_index);
                            stack.push(right_node_index);
                        }
                    }
                    (Some(_), None) => stack.push(left_node_index),
                    (None, Some(_)) => stack.push(right_node_index),
                    (None, None) => {}
                }
            }
        }

        nearest_hit_data
    }
}

pub struct Material {
    pub base_color: Vec3,
    pub metallic_strength: f32,
    pub clear_coat_strength: f32,
    pub clear_coat_roughness: f32,
    pub emissive_color: Vec3,
    pub emissive_strength: f32,
}

impl Material {
    pub fn new(
        base_color: Vec3,
        metallic_strength: f32,
        clear_coat_strength: f32,
        clear_coat_roughness: f32,
        emissive_color: Vec3,
        emissive_strength: f32,
    ) -> Self {
        Material {
            base_color,
            metallic_strength,
            clear_coat_strength,
            clear_coat_roughness,
            emissive_color,
            emissive_strength,
        }
    }
}

pub struct Object {
    pub geometry: Geometry,
    pub material: Material,
}

impl Object {
    pub fn new(geometry: Geometry, material: Material) -> Self {
        Object { geometry, material }
    }
}

pub struct Camera {
    pub fov: f32,
    pub aspect_ratio: f32,
    pub exposure: f32,
    pub position: Vec3,
    pub rotation: Mat3,
}

impl Camera {
    pub fn new(fov: f32, aspect_ratio: f32, exposure: f32, position: Vec3, rotation: Mat3) -> Self {
        Camera {
            fov,
            aspect_ratio,
            exposure,
            position,
            rotation,
        }
    }
}

pub enum Environment {
    Solid {
        base_color: Vec3,
        strength: f32,
    },
    Hdri {
        resolution: (usize, usize),
        buffer: Vec<Vec3>,
    },
}

impl Environment {
    pub fn sample(&self, direction: Vec3) -> Vec3 {
        match self {
            Environment::Solid {
                base_color,
                strength,
            } => base_color * strength,
            Environment::Hdri { resolution, buffer } => {
                let direction: Vec3 = direction.normalize();
                let azimuth: f32 = direction.z.atan2(direction.x) + std::f32::consts::PI;
                let polar: f32 = direction.y.acos();
                let x: usize =
                    (azimuth / std::f32::consts::TAU * (resolution.0 - 1) as f32) as usize;
                let y: usize = (polar / std::f32::consts::PI * (resolution.1 - 1) as f32) as usize;
                buffer[y * resolution.0 + x]
            }
        }
    }
}

pub struct Scene {
    pub objects: Vec<Object>,
    pub environment: Environment,
}

impl Scene {
    pub fn new(objects: Vec<Object>, environment: Environment) -> Self {
        Scene {
            objects,
            environment,
        }
    }

    pub fn intersect_ray(&self, ray: &Ray) -> Option<(f32, Vec3, &Material)> {
        let mut nearest_hit_distance: f32 = f32::INFINITY;
        let mut nearest_hit_data: Option<(f32, Vec3, &Material)> = None;

        for object in &self.objects {
            if let Some((hit_distance, hit_normal)) =
                object.geometry.intersect_ray(ray, nearest_hit_distance)
            {
                nearest_hit_distance = hit_distance;
                nearest_hit_data = Some((hit_distance, hit_normal, &object.material));
            }
        }

        nearest_hit_data
    }
}

pub struct Loader {}

impl Loader {
    pub fn load_obj_file(path: &str) -> Option<Vec<Triangle>> {
        let contents: String = fs::read_to_string(path).ok()?;

        let mut vertices: Vec<Vec3> = Vec::new();
        let mut triangles: Vec<Triangle> = Vec::new();

        for line in contents.lines() {
            let line = line.trim();

            if line.starts_with('v') {
                let parts: Vec<&str> = line.split_whitespace().collect();
                if parts.len() >= 4 {
                    let x: f32 = parts[1].parse::<f32>().ok()?;
                    let y: f32 = parts[2].parse::<f32>().ok()?;
                    let z: f32 = parts[3].parse::<f32>().ok()?;
                    vertices.push(Vec3::new(x, y, z));
                }
            } else if line.starts_with('f') {
                let parts: Vec<&str> = line.split_whitespace().collect();
                if parts.len() >= 4 {
                    let parse_index = |s: &str| s.split('/').next()?.parse::<usize>().ok();

                    let i1: usize = parse_index(parts[1])?;
                    let i2: usize = parse_index(parts[2])?;
                    let i3: usize = parse_index(parts[3])?;
                    let vertex1: Vec3 = *vertices.get(i1 - 1)?;
                    let vertex2: Vec3 = *vertices.get(i2 - 1)?;
                    let vertex3: Vec3 = *vertices.get(i3 - 1)?;

                    triangles.push(Triangle::new(vertex1, vertex2, vertex3));
                }
            }
        }

        Some(triangles)
    }

    pub fn load_hdri_environment(path: &str) -> Option<Environment> {
        let img: DynamicImage = ImageReader::open(path).ok()?.decode().ok()?;
        let rgb32f: image::ImageBuffer<image::Rgb<f32>, Vec<f32>> = match img {
            DynamicImage::ImageRgb32F(img) => img,
            other => other.to_rgb32f(),
        };
        let (width, height) = rgb32f.dimensions();
        let resolution: (usize, usize) = (width as usize, height as usize);
        let buffer: Vec<Vec3> = rgb32f
            .pixels()
            .map(|p: &image::Rgb<f32>| Vec3::new(p[0], p[1], p[2]))
            .collect();
        Some(Environment::Hdri {
            resolution,
            buffer,
        })
    }
}

pub struct Renderer {}

impl Renderer {
    pub fn random_direction(rng: &mut WyRand) -> Vec3 {
        let z: f32 = rng.generate::<f32>() * 2. - 1.;
        let t: f32 = rng.generate::<f32>() * f32::consts::TAU;
        let r: f32 = (1. - z * z).sqrt();
        Vec3::new(r * t.cos(), r * t.sin(), z)
    }

    fn to_srgb(linear: Vec3) -> Vec3 {
        Vec3::new(
            Renderer::linear_to_srgb_component(linear.x),
            Renderer::linear_to_srgb_component(linear.y),
            Renderer::linear_to_srgb_component(linear.z),
        )
    }

    fn linear_to_srgb_component(linear: f32) -> f32 {
        if linear <= 0.0031308 {
            12.92 * linear
        } else {
            1.055 * linear.powf(1. / 2.4) - 0.055
        }
    }

    pub fn tone_map(color: Vec3) -> Vec3 {
        // Narkowicz 2015, "ACES Filmic Tone Mapping Curve"
        let a: f32 = 2.51;
        let b: f32 = 0.03;
        let c: f32 = 2.43;
        let d: f32 = 0.59;
        let e: f32 = 0.14;
        (color * (a * color + b)) / (color * (c * color + d) + e)
    }

    pub fn trace_ray(
        start_ray: &Ray,
        scene: &Scene,
        num_samples: usize,
        num_bounces: usize,
        rng: &mut WyRand,
    ) -> Vec3 {
        let mut ray: Ray;
        let mut incoming_light: Vec3;
        let mut ray_color: Vec3;
        let mut total_incoming_light: Vec3 = Vec3::ZERO;

        for _ in 0..num_samples {
            ray = *start_ray;

            incoming_light = Vec3::ZERO;
            ray_color = Vec3::ONE;

            for _ in 0..num_bounces {
                ray.update_internal_data();

                if let Some((distance, hit_normal, material)) = scene.intersect_ray(&ray) {
                    ray.origin += ray.direction * (distance - 0.0000001);

                    let diffuse_direction: Vec3 =
                        (hit_normal + Renderer::random_direction(rng)).normalize();
                    let specular_direction: Vec3 =
                        ray.direction - 2. * ray.direction.dot(hit_normal) * hit_normal;

                    if rng.generate::<f32>() < material.clear_coat_strength {
                        ray.direction = specular_direction
                            .lerp(diffuse_direction, material.clear_coat_roughness);
                    } else {
                        ray.direction =
                            diffuse_direction.lerp(specular_direction, material.metallic_strength);
                        ray_color *= material.base_color;
                    }

                    incoming_light += material.emissive_color * material.emissive_strength;
                } else {
                    incoming_light += ray_color * scene.environment.sample(ray.direction);

                    break;
                }
            }

            total_incoming_light += incoming_light;
        }

        total_incoming_light / num_samples as f32
    }

    pub fn render(
        resolution: (usize, usize),
        camera: &Camera,
        scene: &Scene,
        num_samples: usize,
        num_bounces: usize,
    ) -> Vec<Vec3> {
        let (resolution_x, resolution_y) = resolution;
        let scale: f32 = (camera.fov * 0.5).tan();
        let exposure_multiplier: f32 = 2.0f32.powf(camera.exposure);
        let mut buffer: Vec<Vec3> = vec![Vec3::ZERO; resolution_x * resolution_y];
        let bar: ProgressBar = ProgressBar::new(resolution_y as u64);
        let start: Instant = Instant::now();

        buffer
            .par_chunks_mut(resolution_x)
            .enumerate()
            .for_each(|(y, row)| {
                let mut rng: WyRand = WyRand::new_seed(y as u64);

                let normalized_screen_position_y: f32 =
                    -((y as f32 + 0.5) / resolution_y as f32 * 2. - 1.);

                for (x, color) in row.iter_mut().enumerate().take(resolution_x) {
                    let normalized_screen_position_x: f32 =
                        (x as f32 + 0.5) / resolution_x as f32 * 2. - 1.;

                    let ray: Ray = Ray::new(
                        camera.position,
                        camera.rotation
                            * Vec3::new(
                                normalized_screen_position_x * camera.aspect_ratio * scale,
                                normalized_screen_position_y * scale,
                                -1.,
                            )
                            .normalize(),
                    );

                    let raw_color: Vec3 =
                        Renderer::trace_ray(&ray, scene, num_samples, num_bounces, &mut rng);
                    let exposed_color: Vec3 = raw_color * exposure_multiplier;
                    let mapped_color: Vec3 = Renderer::tone_map(exposed_color);
                    let final_color: Vec3 =
                        Renderer::to_srgb(mapped_color.clamp(Vec3::ZERO, Vec3::ONE));

                    *color = final_color;
                }

                if y % 10 == 0 {
                    bar.inc(10);
                }
            });

        bar.finish();
        println!("{:.2?}", start.elapsed());

        buffer
    }
}

pub struct Image {}

impl Image {
    pub fn save_render_buffer_to_file(
        resolution: (usize, usize),
        render_buffer: &[Vec3],
        path: &str,
    ) -> Result<(), ImageError> {
        let buffer: Vec<u8> = render_buffer
            .iter()
            .flat_map(|float_value| {
                [
                    (float_value.x * 255.) as u8,
                    (float_value.y * 255.) as u8,
                    (float_value.z * 255.) as u8,
                ]
            })
            .collect();

        save_buffer(
            path,
            &buffer,
            resolution.0 as u32,
            resolution.1 as u32,
            ExtendedColorType::Rgb8,
        )
    }
}

fn main() {
    let resolution: (usize, usize) = (1920, 1080);
    let camera: Camera = Camera::new(
        1.2,
        16. / 9.,
        -2.,
        Vec3::new(0., 0., 1.),
        Mat3::from_euler(glam::EulerRot::XYZ, 0., 0., 0.),
    );
    let dragon_obj_triangles: Vec<Triangle> =
        Loader::load_obj_file("res/dragon.obj").expect("Failed to load obj file");
    let plane_obj_triangles: Vec<Triangle> =
        Loader::load_obj_file("res/plane.obj").expect("Failed to load obj file");
    let objects: Vec<Object> = vec![
        Object::new(
            Geometry::new(
                &dragon_obj_triangles,
                Vec3::ZERO,
                Vec3::ONE,
                Mat3::from_euler(glam::EulerRot::XYZ, 0., 2., 0.),
            ),
            Material::new(Vec3::splat(0.5), 0.5, 0., 0., Vec3::ZERO, 0.),
        ),
        Object::new(
            Geometry::new(
                &plane_obj_triangles,
                Vec3::new(0., -0.275, 0.),
                Vec3::splat(3.),
                Mat3::from_euler(glam::EulerRot::XYZ, 0., 0., 0.),
            ),
            Material::new(Vec3::splat(1.), 0., 0., 0., Vec3::ZERO, 0.),
        ),
    ];
    let environment: Environment =
        Loader::load_hdri_environment("res/sky.hdr").expect("Failed to load hdri environment");
    let scene: Scene = Scene::new(objects, environment);
    let render_buffer: Vec<Vec3> = Renderer::render(resolution, &camera, &scene, 256, 6);
    Image::save_render_buffer_to_file(resolution, &render_buffer, "render.png")
        .expect("Failed to save render buffer to file");
}
