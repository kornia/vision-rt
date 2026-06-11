mod node;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    bubbaloop_node::run_node::<node::TrtDetectorNode>().await
}
