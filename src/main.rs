use glam::{EulerRot, Mat3A, Vec3A};

pub mod math {
    use glam::Vec3A;

    #[derive(Clone, Copy)]
    pub struct Ray {
        pub origin: Vec3A,
        pub direction: Vec3A,
        pub inverse_direction: Vec3A,
    }

    impl Ray {
        pub fn new(origin: Vec3A, direction: Vec3A) -> Self {
            Self {
                origin,
                direction,
                inverse_direction: Vec3A::ZERO,
            }
        }

        pub fn update_internal_data(&mut self) {
            self.inverse_direction = Vec3A::ONE / self.direction;
        }
    }

    pub struct BoundingBox {
        pub(crate) min: Vec3A,
        pub(crate) max: Vec3A,
    }

    impl BoundingBox {
        pub fn new(min: Vec3A, max: Vec3A) -> Self {
            Self { min, max }
        }

        pub fn empty() -> Self {
            Self {
                min: Vec3A::INFINITY,
                max: Vec3A::NEG_INFINITY,
            }
        }

        pub fn expand_to_bounds(&mut self, min: Vec3A, max: Vec3A) {
            self.min = self.min.min(min);
            self.max = self.max.max(max);
        }

        pub fn expand_to_point(&mut self, point: Vec3A) {
            self.min = self.min.min(point);
            self.max = self.max.max(point);
        }

        #[inline(always)]
        pub fn intersect(&self, ray: &Ray, nearest_hit_distance: f32) -> Option<f32> {
            let ray_origin: Vec3A = ray.origin;
            let ray_inverse_direction: Vec3A = ray.inverse_direction;
            let t0: Vec3A = (self.min - ray_origin) * ray_inverse_direction;
            let t1: Vec3A = (self.max - ray_origin) * ray_inverse_direction;
            let tmin_v: Vec3A = t0.min(t1);
            let tmax_v: Vec3A = t0.max(t1);
            let tmin: f32 = tmin_v.x.max(tmin_v.y).max(tmin_v.z).max(0.0);
            (tmin < nearest_hit_distance && tmax_v.x.min(tmax_v.y).min(tmax_v.z) >= tmin)
                .then_some(tmin)
        }
    }

    pub struct Triangle {
        pub v0: Vec3A,
        pub v1: Vec3A,
        pub v2: Vec3A,
        pub material_index: usize,
        pub u: Vec3A,
        pub v: Vec3A,
        pub normal: Vec3A,
        pub uu: f32,
        pub uv: f32,
        pub vv: f32,
        pub d0: f32,
        pub inv_d1: f32,
    }

    impl Triangle {
        pub fn new(v0: Vec3A, v1: Vec3A, v2: Vec3A, material_index: usize) -> Self {
            Self {
                v0,
                v1,
                v2,
                material_index,
                u: Vec3A::ZERO,
                v: Vec3A::ZERO,
                normal: Vec3A::ZERO,
                uu: 0.,
                uv: 0.,
                vv: 0.,
                d0: 0.,
                inv_d1: 0.,
            }
        }

        pub fn update_internal_data(&mut self) {
            self.u = self.v1 - self.v0;
            self.v = self.v2 - self.v0;
            self.uu = self.u.length_squared();
            self.uv = self.u.dot(self.v);
            self.vv = self.v.length_squared();
            self.normal = self.u.cross(self.v).normalize();
            self.d0 = -self.v0.dot(self.normal);
            self.inv_d1 = 1. / (self.uu * self.vv - self.uv * self.uv);
        }

        pub fn min(&self) -> Vec3A {
            self.v0.min(self.v1).min(self.v2)
        }

        pub fn max(&self) -> Vec3A {
            self.v0.max(self.v1).max(self.v2)
        }

        pub fn centroid(&self) -> Vec3A {
            (self.v0 + self.v1 + self.v2) * (1. / 3.)
        }

        pub fn intersect(&self, ray: &Ray, nearest_hit_distance: f32) -> Option<f32> {
            let den: f32 = self.normal.dot(ray.direction);
            if den >= 0. {
                return None;
            }
            let distance: f32 = -(self.normal.dot(ray.origin) + self.d0) / den;
            if distance < 0. || distance > nearest_hit_distance {
                return None;
            }
            let w: Vec3A = ray.origin + ray.direction * distance - self.v0;
            let wu: f32 = w.dot(self.u);
            let wv: f32 = w.dot(self.v);
            let s: f32 = (self.vv * wu - self.uv * wv) * self.inv_d1;
            let t: f32 = (self.uu * wv - self.uv * wu) * self.inv_d1;
            if s < 0. || t < 0. || (s + t) > 1. {
                return None;
            }
            Some(distance)
        }
    }
}

pub mod bvh {
    use std::time::Instant;

    use glam::Vec3A;

    use crate::math::{BoundingBox, Ray, Triangle};

    pub struct BVH {
        nodes_section_a: Vec<(usize, usize, bool)>,
        nodes_section_b: Vec<BoundingBox>,
        root_node_index: usize,
    }

    impl BVH {
        pub fn build(input_triangles: Vec<Triangle>) -> (Self, Vec<Triangle>) {
            let mut nodes_section_b: Vec<BoundingBox> = Vec::new();
            let mut nodes_section_a: Vec<(usize, usize, bool)> = Vec::new();
            let mut final_triangles: Vec<Triangle> = Vec::with_capacity(input_triangles.len());

            let start_time: Instant = Instant::now();

            let root_node_index: usize = Self::split(
                input_triangles,
                &mut nodes_section_a,
                &mut nodes_section_b,
                &mut final_triangles,
                0,
            );

            final_triangles
                .iter_mut()
                .for_each(|triangle| triangle.update_internal_data());

            println!("bvh build: {:#?}", start_time.elapsed());

            (
                Self {
                    nodes_section_a,
                    nodes_section_b,
                    root_node_index,
                },
                final_triangles,
            )
        }

        fn split(
            input_triangles: Vec<Triangle>,
            nodes_section_a: &mut Vec<(usize, usize, bool)>,
            nodes_section_b: &mut Vec<BoundingBox>,
            final_triangles: &mut Vec<Triangle>,
            depth: usize,
        ) -> usize {
            let input_triangles_len: usize = input_triangles.len();
            let mut node_bounding_box: BoundingBox = BoundingBox::empty();
            let mut centroid_bounds: BoundingBox = BoundingBox::empty();

            for tri in &input_triangles {
                node_bounding_box.expand_to_bounds(tri.min(), tri.max());
                centroid_bounds.expand_to_point(tri.centroid());
            }

            let node_bb_size: Vec3A = node_bounding_box.max - node_bounding_box.min;
            let node_bb_area: f32 = 2.0
                * (node_bb_size.x * node_bb_size.y
                    + node_bb_size.x * node_bb_size.z
                    + node_bb_size.y * node_bb_size.z);

            if input_triangles_len <= 6 || depth >= 128 {
                let start: usize = final_triangles.len();
                final_triangles.extend(input_triangles);
                let end: usize = final_triangles.len();

                nodes_section_b.push(node_bounding_box);
                nodes_section_a.push((start, end, true));
            } else {
                let centroid_extent: Vec3A = centroid_bounds.max - centroid_bounds.min;
                let centroid_center: Vec3A = (centroid_bounds.min + centroid_bounds.max) * 0.5;

                let mut best_cost: f32 = f32::INFINITY;
                let mut best_sides: Option<Vec<bool>> = None;

                let mut left_indices: Vec<usize> = Vec::new();
                let mut right_indices: Vec<usize> = Vec::new();

                let test_count: i32 = 7;
                let half_tests: f32 = (test_count - 1) as f32 / 2.0;

                for axis in 0..3 {
                    if centroid_extent[axis] <= 1e-6 {
                        continue;
                    }

                    for test in 0..test_count {
                        left_indices.clear();
                        right_indices.clear();

                        let offset = (test as f32 - half_tests) / half_tests;
                        let split_position =
                            centroid_center[axis] + offset * centroid_extent[axis] * 0.75;

                        for (i, tri) in input_triangles.iter().enumerate() {
                            if tri.centroid()[axis] < split_position {
                                left_indices.push(i);
                            } else {
                                right_indices.push(i);
                            }
                        }

                        if left_indices.is_empty() || right_indices.is_empty() {
                            continue;
                        }

                        let mut left_bb: BoundingBox = BoundingBox::empty();
                        let mut right_bb: BoundingBox = BoundingBox::empty();

                        for &i in &left_indices {
                            let t = &input_triangles[i];
                            left_bb.expand_to_bounds(t.min(), t.max());
                        }
                        for &i in &right_indices {
                            let t = &input_triangles[i];
                            right_bb.expand_to_bounds(t.min(), t.max());
                        }

                        let lsize: Vec3A = left_bb.max - left_bb.min;
                        let rsize: Vec3A = right_bb.max - right_bb.min;

                        let left_area: f32 =
                            2.0 * (lsize.x * lsize.y + lsize.x * lsize.z + lsize.y * lsize.z);
                        let right_area: f32 =
                            2.0 * (rsize.x * rsize.y + rsize.x * rsize.z + rsize.y * rsize.z);

                        let cost: f32 = (left_area / node_bb_area) * (left_indices.len() as f32)
                            + (right_area / node_bb_area) * (right_indices.len() as f32);

                        if cost < best_cost {
                            let mut sides: Vec<bool> = vec![false; input_triangles_len];
                            for &i in &right_indices {
                                sides[i] = true;
                            }
                            best_cost = cost;
                            best_sides = Some(sides);
                        }
                    }
                }

                match best_sides {
                    Some(best_sides) => {
                        let mut left_tris: Vec<Triangle> = Vec::new();
                        let mut right_tris: Vec<Triangle> = Vec::new();

                        for (side, tri) in best_sides.into_iter().zip(input_triangles) {
                            if side {
                                right_tris.push(tri);
                            } else {
                                left_tris.push(tri);
                            }
                        }

                        let left_index: usize = Self::split(
                            left_tris,
                            nodes_section_a,
                            nodes_section_b,
                            final_triangles,
                            depth + 1,
                        );
                        let right_index: usize = Self::split(
                            right_tris,
                            nodes_section_a,
                            nodes_section_b,
                            final_triangles,
                            depth + 1,
                        );

                        nodes_section_b.push(node_bounding_box);
                        nodes_section_a.push((left_index, right_index, false));
                    }
                    None => {
                        let start: usize = final_triangles.len();
                        final_triangles.extend(input_triangles);
                        let end: usize = final_triangles.len();

                        nodes_section_b.push(node_bounding_box);
                        nodes_section_a.push((start, end, true));
                    }
                };
            }

            nodes_section_a.len() - 1
        }
    }

    pub struct BVHIntersector<'a> {
        bvh: &'a BVH,
        stack: Vec<usize>,
    }

    impl<'a> BVHIntersector<'a> {
        pub fn new(bvh: &'a BVH) -> Self {
            Self {
                bvh,
                stack: Vec::with_capacity(128),
            }
        }

        pub fn intersect(&mut self, ray: &Ray, triangles: &[Triangle]) -> Option<(f32, usize)> {
            let mut nearest_hit_distance: f32 = f32::INFINITY;
            let mut nearest_hit_data: Option<(f32, usize)> = None;
            let nodes_section_a: &[(usize, usize, bool)] = &self.bvh.nodes_section_a;
            let nodes_section_b: &[BoundingBox] = &self.bvh.nodes_section_b;

            let _ =
                nodes_section_b[self.bvh.root_node_index].intersect(ray, nearest_hit_distance)?;

            self.stack.clear();
            self.stack.push(self.bvh.root_node_index);

            while let Some(node_index) = self.stack.pop() {
                let (a, b, is_leaf) = nodes_section_a[node_index];

                if !is_leaf {
                    match (
                        nodes_section_b[a].intersect(ray, nearest_hit_distance),
                        nodes_section_b[b].intersect(ray, nearest_hit_distance),
                    ) {
                        (None, None) => {}
                        (None, Some(_)) => self.stack.push(b),
                        (Some(_), None) => self.stack.push(a),
                        (Some(left_hit_distance), Some(right_hit_distance)) => {
                            if left_hit_distance < right_hit_distance {
                                self.stack.push(b);
                                self.stack.push(a);
                            } else {
                                self.stack.push(a);
                                self.stack.push(b);
                            };
                        }
                    }
                } else {
                    for triangle_index in a..b {
                        let triangle: &Triangle = &triangles[triangle_index];

                        if let Some(hit_distance) = triangle.intersect(ray, nearest_hit_distance) {
                            nearest_hit_distance = hit_distance;
                            nearest_hit_data = Some((hit_distance, triangle_index));
                        }
                    }
                }
            }

            nearest_hit_data
        }
    }
}

pub mod camera {
    use glam::{Mat3A, Vec3A};

    pub struct Camera {
        pub fov: f32,
        pub aspect_ratio: f32,
        pub focus_distance: f32,
        pub lens_radius: f32,
        pub position: Vec3A,
        pub rotation: Mat3A,
    }

    impl Camera {
        pub fn new(fov: f32, aspect_ratio: f32, focus_distance: f32, lens_radius: f32, position: Vec3A, rotation: Mat3A) -> Self {
            Camera {
                fov,
                aspect_ratio,
                focus_distance,
                lens_radius,
                position,
                rotation,
            }
        }
    }
}

pub mod obj_loader {
    use crate::math::Triangle;
    use glam::{Mat3A, Vec3A};
    use std::fs;

    pub fn load_obj_from_file(
        path: &str,
        position: Vec3A,
        scale: Vec3A,
        rotation: Mat3A,
        material_index: usize,
    ) -> Option<Vec<Triangle>> {
        let contents: String = fs::read_to_string(path).ok()?;

        let mut vertices: Vec<Vec3A> = Vec::new();
        let mut triangles: Vec<Triangle> = Vec::new();

        for line in contents.lines() {
            let line: &str = line.trim();
            let parts: Vec<&str> = line.split_whitespace().collect();

            if parts.len() == 4 {
                if parts[0] == "v" {
                    vertices.push(Vec3A::new(
                        parts[1].parse::<f32>().ok()?,
                        parts[2].parse::<f32>().ok()?,
                        parts[3].parse::<f32>().ok()?,
                    ));
                } else if parts[0] == "f" {
                    triangles.push(Triangle::new(
                        rotation
                            * *vertices
                                .get(parts[1].split('/').next()?.parse::<usize>().ok()? - 1)?
                            * scale
                            + position,
                        rotation
                            * *vertices
                                .get(parts[2].split('/').next()?.parse::<usize>().ok()? - 1)?
                            * scale
                            + position,
                        rotation
                            * *vertices
                                .get(parts[3].split('/').next()?.parse::<usize>().ok()? - 1)?
                            * scale
                            + position,
                        material_index,
                    ));
                }
            }
        }

        Some(triangles)
    }
}

pub mod scene {
    use glam::Vec3A;
    use image::{DynamicImage, ImageReader};

    use crate::math::Triangle;

    pub enum Material {
        Debug,
        Emissive { albedo: Vec3A, strength: f32 },
    }

    pub enum Environment {
        Solid {
            base_color: Vec3A,
            strength: f32,
        },
        Hdri {
            resolution: (usize, usize),
            buffer: Vec<Vec3A>,
        },
    }

    impl Environment {
        pub fn load_hdri_environment_from_file(path: &str, strength: f32) -> Option<Environment> {
            let img: DynamicImage = ImageReader::open(path).ok()?.decode().ok()?;
            let rgb32f: image::ImageBuffer<image::Rgb<f32>, Vec<f32>> = match img {
                DynamicImage::ImageRgb32F(img) => img,
                other => other.to_rgb32f(),
            };
            let (width, height) = rgb32f.dimensions();
            let resolution: (usize, usize) = (width as usize, height as usize);
            let buffer: Vec<Vec3A> = rgb32f
                .pixels()
                .map(|p: &image::Rgb<f32>| {
                    Vec3A::new(p[0] * strength, p[1] * strength, p[2] * strength)
                })
                .collect();
            Some(Environment::Hdri { resolution, buffer })
        }

        pub fn sample(&self, direction: &Vec3A) -> Vec3A {
            match self {
                Environment::Solid {
                    base_color,
                    strength,
                } => base_color * strength,
                Environment::Hdri { resolution, buffer } => {
                    let direction: Vec3A = direction.normalize();
                    let azimuth: f32 = direction.z.atan2(direction.x) + std::f32::consts::PI;
                    let polar: f32 = direction.y.acos();
                    let x: usize =
                        (azimuth / std::f32::consts::TAU * (resolution.0 - 1) as f32) as usize;
                    let y: usize =
                        (polar / std::f32::consts::PI * (resolution.1 - 1) as f32) as usize;
                    buffer[y * resolution.0 + x]
                }
            }
        }
    }

    pub struct Scene {
        pub triangles: Vec<Triangle>,
        pub environment: Environment,
        pub materials: Vec<Material>,
    }
}

pub mod renderer {
    use std::time::Instant;

    use glam::Vec3A;
    use indicatif::ProgressBar;
    use nanorand::{Rng, WyRand};
    use rayon::{
        iter::{IndexedParallelIterator, ParallelIterator},
        slice::ParallelSliceMut,
    };

    use crate::{
        bvh::{BVH, BVHIntersector},
        camera::Camera,
        math::{Ray, Triangle},
        scene::{Material, Scene},
    };

    fn sample_sphere(rng: &mut WyRand) -> Vec3A {
        let z: f32 = rng.generate::<f32>() * 2. - 1.;
        let t: f32 = rng.generate::<f32>() * std::f32::consts::TAU;
        let r: f32 = (1. - z * z).sqrt();
        Vec3A::new(r * t.cos(), r * t.sin(), z)
    }

    fn sample_disk(rng: &mut WyRand) -> Vec3A {
        let u1: f32 = rng.generate::<f32>() * 2.0 - 1.0;
        let u2: f32 = rng.generate::<f32>() * 2.0 - 1.0;

        if u1 == 0.0 && u2 == 0.0 {
            return Vec3A::ZERO;
        }

        let (r, theta) = if u1.abs() > u2.abs() {
            (u1, std::f32::consts::FRAC_PI_4 * (u2 / u1))
        } else {
            (
                u2,
                std::f32::consts::FRAC_PI_2 - std::f32::consts::FRAC_PI_4 * (u1 / u2),
            )
        };

        Vec3A::new(r * theta.cos(), r * theta.sin(), 0.0)
    }

    pub fn render(
        resolution: (usize, usize),
        camera: &Camera,
        scene: &Scene,
        bvh: &BVH,
        num_samples: usize,
        num_bounces: usize,
    ) -> Vec<Vec3A> {
        let (resolution_x, resolution_y) = resolution;
        let inverse_resolution_x_2: f32 = 1. / resolution_x as f32 * 2.;
        let inverse_resolution_y_2: f32 = 1. / resolution_y as f32 * 2.;
        let scale: f32 = (camera.fov * 0.5).tan();
        let camera_rotation_x: Vec3A = camera.rotation * Vec3A::X;
        let camera_rotation_y: Vec3A = camera.rotation * Vec3A::Y;
        let camera_rotation_z: Vec3A = camera.rotation * Vec3A::Z;
        let mut buffer: Vec<Vec3A> = vec![Vec3A::ZERO; resolution_x * resolution_y];
        let reciprocal_num_samples: f32 = 1. / num_samples as f32;
        let scene_triangles: &[Triangle] = &scene.triangles;
        let scene_materials: &[Material] = &scene.materials;

        let bar: ProgressBar = ProgressBar::new(resolution_y as u64);
        let start_time: Instant = Instant::now();

        buffer
            .par_chunks_mut(resolution_x)
            .enumerate()
            .for_each(|(y, row)| {
                let mut rng: WyRand = WyRand::new_seed(y as u64);
                let mut bvh_intersector: BVHIntersector<'_> = BVHIntersector::new(bvh);

                let normalized_screen_position_y: f32 =
                    -((y as f32 + 0.5) * inverse_resolution_y_2 - 1.);

                for x in 0..resolution_x {
                    let normalized_screen_position_x: f32 =
                        (x as f32 + 0.5) * inverse_resolution_x_2 - 1.;

                    let base_ray: Ray = Ray::new(
                        camera.position,
                        (camera_rotation_x
                            * normalized_screen_position_x
                            * camera.aspect_ratio
                            * scale
                            + camera_rotation_y * normalized_screen_position_y * scale
                            + camera_rotation_z * -1.)
                            .normalize(),
                    );

                    let focus_point: Vec3A = base_ray.origin + base_ray.direction * camera.focus_distance;

                    let mut total_incoming_light: Vec3A = Vec3A::ZERO;

                    for _ in 0..num_samples {
                        let mut ray: Ray = base_ray;

                        let jitter: Vec3A = sample_disk(&mut rng);
                        let lens_offset: Vec3A = camera_rotation_x * jitter.x * camera.lens_radius
                            + camera_rotation_y * jitter.y * camera.lens_radius;

                        ray.origin = camera.position + lens_offset;
                        ray.direction = (focus_point - ray.origin).normalize();

                        let mut incoming_light: Vec3A = Vec3A::ZERO;
                        let mut ray_color: Vec3A = Vec3A::ONE;

                        for _ in 0..=num_bounces {
                            ray.update_internal_data();

                            if let Some((hit_distance, triangle_index)) =
                                bvh_intersector.intersect(&ray, scene_triangles)
                            {
                                let triangle: &Triangle = &scene_triangles[triangle_index];

                                match &scene_materials[triangle.material_index] {
                                    Material::Debug => {
                                        ray.origin += ray.direction * (hit_distance - 0.0000001);
                                        ray.direction =
                                            (triangle.normal + sample_sphere(&mut rng)).normalize();

                                        ray_color *= 0.75;
                                    }
                                    Material::Emissive { albedo, strength } => {
                                        incoming_light += ray_color * albedo * strength;
                                        break;
                                    }
                                }
                            } else {
                                incoming_light +=
                                    ray_color * scene.environment.sample(&ray.direction);
                                break;
                            }
                        }

                        total_incoming_light += incoming_light;
                    }

                    row[x] = total_incoming_light * reciprocal_num_samples;
                }

                if y % 10 == 9 {
                    bar.inc(10);
                }
            });

        bar.finish();
        println!("render: {:#?}", start_time.elapsed());

        buffer
    }
}

pub mod color_management {
    use std::time::Instant;

    use glam::Vec3A;
    use rayon::iter::{IntoParallelRefMutIterator, ParallelIterator};

    pub fn apply_exposure(color: &mut Vec3A, exposure_value: f32) {
        let multiplier_strength: f32 = exposure_value.exp2();

        *color *= multiplier_strength;
    }

    pub enum ViewTransform {
        None,
        Aces,
    }

    impl ViewTransform {
        fn aces(color: &Vec3A) -> Vec3A {
            const A: f32 = 2.51;
            const B: f32 = 0.03;
            const C: f32 = 2.43;
            const D: f32 = 0.59;
            const E: f32 = 0.14;
            (color * (A * color + B)) / (color * (C * color + D) + E)
        }

        pub fn apply(&self, color: &mut Vec3A) {
            match self {
                ViewTransform::None => {}
                ViewTransform::Aces => *color = Self::aces(color),
            }
        }
    }

    pub enum Colorspace {
        None,
        Srgb,
    }

    impl Colorspace {
        fn srgb_component(linear: f32) -> f32 {
            if linear <= 0.0031308 {
                12.92 * linear
            } else {
                1.055 * linear.powf(1. / 2.4) - 0.055
            }
        }

        fn srgb(color: &Vec3A) -> Vec3A {
            Vec3A::new(
                Self::srgb_component(color.x),
                Self::srgb_component(color.y),
                Self::srgb_component(color.z),
            )
        }

        pub fn apply(&self, color: &mut Vec3A) {
            match self {
                Colorspace::None => {}
                Colorspace::Srgb => *color = Self::srgb(color),
            }
        }
    }

    pub struct ColorManager {
        pub exposure_value: f32,
        pub view_transform: ViewTransform,
        pub colorspace: Colorspace,
    }

    impl ColorManager {
        pub fn apply(&self, buffer: &mut Vec<Vec3A>) {
            let start_time: Instant = Instant::now();
            buffer.par_iter_mut().for_each(|color| {
                apply_exposure(color, self.exposure_value);
                self.view_transform.apply(color);
                self.colorspace.apply(color);
            });
            println!("color management: {:#?}", start_time.elapsed());
        }
    }
}

pub mod image {
    use glam::Vec3A;
    use image::{self, ExtendedColorType, ImageError, save_buffer};
    use rayon::iter::{IntoParallelRefIterator, ParallelIterator};

    #[inline]
    fn to_u8(v: f32) -> u8 {
        (v.clamp(0.0, 1.0) * 255.0 + 0.5) as u8
    }

    pub fn save_buffer_to_file(
        resolution: (usize, usize),
        buffer: &[Vec3A],
        path: &str,
    ) -> Result<(), ImageError> {
        let byte_buffer: Vec<u8> = buffer
            .par_iter()
            .map(|c| [to_u8(c.x), to_u8(c.y), to_u8(c.z)])
            .collect::<Vec<[u8; 3]>>()
            .iter()
            .flatten()
            .copied()
            .collect();

        save_buffer(
            path,
            &byte_buffer,
            resolution.0 as u32,
            resolution.1 as u32,
            ExtendedColorType::Rgb8,
        )
    }
}

fn main() {
    let resolution: (usize, usize) = (1920, 1080);

    let camera: camera::Camera = camera::Camera::new(
        1.,
        16. / 9.,
        1.,
        0.04,
        Vec3A::new(0., 0., 1.),
        Mat3A::from_euler(EulerRot::XYZ, 0., 0., 0.),
    );

    let materials: Vec<scene::Material> = vec![scene::Material::Debug];
    let environment: scene::Environment =
        scene::Environment::load_hdri_environment_from_file("res/sky.hdr", 0.2)
            .expect("Failed to load hdr file");

    let mut triangles: Vec<math::Triangle> = Vec::new();
    triangles.extend(
        obj_loader::load_obj_from_file(
            "res/dragon.obj",
            Vec3A::new(0., 0., 0.),
            Vec3A::ONE,
            Mat3A::from_euler(EulerRot::XYZ, 0., 2., 0.),
            0,
        )
        .expect("Failed to load obj file"),
    );
    triangles.extend(
        obj_loader::load_obj_from_file(
            "res/plane.obj",
            Vec3A::new(0., -0.275, 0.),
            Vec3A::splat(3.),
            Mat3A::from_euler(EulerRot::XYZ, 0., 0., 0.),
            0,
        )
        .expect("Failed to load obj file"),
    );

    let (bvh, triangles) = bvh::BVH::build(triangles);

    let scene: scene::Scene = scene::Scene {
        triangles,
        environment,
        materials,
    };

    let color_manager: color_management::ColorManager = color_management::ColorManager {
        exposure_value: 0.,
        view_transform: color_management::ViewTransform::Aces,
        colorspace: color_management::Colorspace::Srgb,
    };

    let mut buffer: Vec<Vec3A> = renderer::render(resolution, &camera, &scene, &bvh, 16, 6);

    color_manager.apply(&mut buffer);

    image::save_buffer_to_file(resolution, &buffer, "render.png")
        .expect("Failed to save buffer to file");
}
