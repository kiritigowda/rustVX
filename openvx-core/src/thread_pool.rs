use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{mpsc, Arc, Mutex};
use std::thread;

/// Simple fixed-size thread pool for executing node tasks in parallel.
/// Created on-demand via a static OnceLock; size defaults to hardware
/// concurrency or 4 if unavailable.

pub struct ThreadPool {
    workers: Vec<Worker>,
    sender: Option<mpsc::Sender<Job>>,
}

type Job = Box<dyn FnOnce() + Send + 'static>;

impl ThreadPool {
    /// Create a new ThreadPool with the given number of threads.
    pub fn new(size: usize) -> ThreadPool {
        assert!(size > 0);
        let (sender, receiver) = mpsc::channel::<Job>();
        let receiver = Arc::new(Mutex::new(receiver));
        let mut workers = Vec::with_capacity(size);
        for _ in 0..size {
            workers.push(Worker::new(Arc::clone(&receiver)));
        }
        ThreadPool {
            workers,
            sender: Some(sender),
        }
    }

    /// Execute a job on one of the pool threads.
    pub fn execute<F>(&self, f: F)
    where
        F: FnOnce() + Send + 'static,
    {
        let job = Box::new(f);
        if let Some(ref sender) = self.sender {
            sender.send(job).expect("ThreadPool sender disconnected");
        }
    }
}

impl Drop for ThreadPool {
    fn drop(&mut self) {
        // Drop the sender so workers exit
        drop(self.sender.take());
        for worker in &mut self.workers {
            if let Some(thread) = worker.thread.take() {
                thread.join().expect("Worker thread panicked");
            }
        }
    }
}

struct Worker {
    thread: Option<thread::JoinHandle<()>>,
}

impl Worker {
    fn new(receiver: Arc<Mutex<mpsc::Receiver<Job>>>) -> Worker {
        let thread = thread::spawn(move || {
            loop {
                let message = receiver.lock().unwrap().recv();
                match message {
                    Ok(job) => job(),
                    Err(_) => break,
                }
            }
        });
        Worker {
            thread: Some(thread),
        }
    }
}

/// Global thread pool, lazily initialized.
use std::sync::OnceLock;

static GLOBAL_POOL: OnceLock<Arc<ThreadPool>> = OnceLock::new();

/// Initialize the global thread pool with the given size.
/// Safe to call multiple times — only the first call takes effect.
pub fn init_global_pool(size: usize) {
    let _ = GLOBAL_POOL.get_or_init(|| Arc::new(ThreadPool::new(size)));
}

/// Get a reference to the global thread pool, initializing with default
/// size (hardware concurrency or 4) if not already set.
pub fn get_global_pool() -> Option<Arc<ThreadPool>> {
    let pool = GLOBAL_POOL.get_or_init(|| {
        let size = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4);
        Arc::new(ThreadPool::new(size))
    });
    Some(Arc::clone(pool))
}

/// Set a custom pool size (for testing or tuning). Returns false if pool
/// already initialized.
pub fn set_pool_size(size: usize) -> bool {
    if GLOBAL_POOL.get().is_some() {
        return false;
    }
    let _ = GLOBAL_POOL.set(Arc::new(ThreadPool::new(size)));
    true
}
