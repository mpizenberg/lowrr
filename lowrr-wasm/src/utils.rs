// SPDX-License-Identifier: MPL-2.0

use log::{Level, Metadata, Record, SetLoggerError};
use wasm_bindgen::prelude::*;

pub fn set_panic_hook() {
    // When the `console_error_panic_hook` feature is enabled, we can call the
    // `set_panic_hook` function at least once during initialization, and then
    // we will get better error messages if our code ever panics.
    //
    // For more details see
    // https://github.com/rustwasm/console_error_panic_hook#readme
    #[cfg(feature = "console_error_panic_hook")]
    console_error_panic_hook::set_once();
}

#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_namespace = console)]
    pub fn log(s: &str);
}

#[wasm_bindgen(raw_module = "../worker.mjs")]
extern "C" {
    fn appLog(level: u32, content: &str);
}

// Macro console_log! similar to println!
macro_rules! console_log {
    ($($t:tt)*) => (crate::utils::log(&format_args!($($t)*).to_string()))
}

// Log implementation

#[derive(Clone, Copy)]
pub struct WasmLogger {
    max_level: Level,
}

impl WasmLogger {
    pub fn init(max_level: Level) -> Result<Self, SetLoggerError> {
        let wasm_logger = WasmLogger { max_level };
        log::set_boxed_logger(Box::new(wasm_logger))
            .map(|_| log::set_max_level(max_level.to_level_filter()))?;
        Ok(wasm_logger)
    }
}

impl log::Log for WasmLogger {
    fn enabled(&self, metadata: &Metadata) -> bool {
        metadata.level() <= self.max_level
    }

    fn log(&self, record: &Record) {
        if self.enabled(record.metadata()) {
            appLog(level_u32(record.level()), &record.args().to_string());
        }
    }

    fn flush(&self) {}
}

fn level_u32(level: Level) -> u32 {
    match level {
        Level::Error => 0,
        Level::Warn => 1,
        Level::Info => 2,
        Level::Debug => 3,
        Level::Trace => 4,
    }
}
