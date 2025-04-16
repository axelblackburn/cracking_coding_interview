use data_structures;

pub fn hello_from_leet() -> String {
    let message = data_structures::hello_from_data_structures();
    format!("Hello from leet!  {}", message)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        let result = hello_from_leet();
        assert_eq!(result, "Hello from leet!  Hello from data_structures!");
    }
}
