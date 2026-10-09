//! Command-line arguments of the form `--name value`, and bare `--flag`s.

use std::collections::HashMap;
use std::str::FromStr;

pub struct Args {
    values: HashMap<String, String>,
    flags: Vec<String>,
}

impl Args {
    /// The process's arguments. A `--name` followed by a value that doesn't start with `--` takes it, and any other
    /// `--name` is a flag.
    pub fn parse() -> Args {
        Args::from(std::env::args().skip(1).collect())
    }

    pub fn from(args: Vec<String>) -> Args {
        let (mut values, mut flags) = (HashMap::new(), Vec::new());
        let mut it = args.into_iter().peekable();

        while let Some(arg) = it.next() {
            let Some(name) = arg.strip_prefix("--") else {
                fail(&format!("unexpected argument {arg}"));
            };
            match it.next_if(|next| !next.starts_with("--")) {
                Some(value) => _ = values.insert(name.to_string(), value),
                None => flags.push(name.to_string()),
            }
        }

        Args { values, flags }
    }

    pub fn flag(&self, name: &str) -> bool {
        self.flags.iter().any(|f| f == name)
    }

    pub fn str(&self, name: &str) -> Option<&str> {
        self.values.get(name).map(String::as_str)
    }

    /// The value of `--name`, which must be there.
    pub fn need(&self, name: &str) -> &str {
        self.str(name).unwrap_or_else(|| fail(&format!("--{name} is required")))
    }

    /// The value of `--name` parsed, or `default` when it isn't there.
    pub fn get<T: FromStr>(&self, name: &str, default: T) -> T {
        match self.str(name) {
            Some(v) => v
                .parse()
                .unwrap_or_else(|_| fail(&format!("--{name} has a bad value: {v}"))),
            None => default,
        }
    }

    /// A comma-separated list, eg. `--nprobe 128,160,192`.
    pub fn list<T: FromStr + Clone>(&self, name: &str, default: &[T]) -> Vec<T> {
        match self.str(name) {
            Some(v) => v
                .split(',')
                .map(|x| {
                    x.trim()
                        .parse()
                        .unwrap_or_else(|_| fail(&format!("--{name} has a bad value: {x}")))
                })
                .collect(),
            None => default.to_vec(),
        }
    }

    /// Every argument again, for a child process that runs one trial.
    pub fn forward(&self) -> Vec<String> {
        let mut out = Vec::new();

        for (name, value) in &self.values {
            out.push(format!("--{name}"));
            out.push(value.clone());
        }

        out.extend(self.flags.iter().map(|f| format!("--{f}")));
        out
    }
}

/// Prints `message` and exits with status 2, for bad arguments.
pub fn fail(message: &str) -> ! {
    eprintln!("error: {message}");
    std::process::exit(2)
}
