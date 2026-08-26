KEY_DIR=".devcontainer/multi-node"
PRIVATE_KEY="$KEY_DIR/id_ed25519"
PUBLIC_KEY="${PRIVATE_KEY}.pub"

if [ -f "$PRIVATE_KEY" ] && [ -f "$PUBLIC_KEY" ]; then
    echo "Both the private and public keys already exist. Skipping key generation..."
else
    echo "One or both of the public/private keys are missing. Generating new keys..."
    ssh-keygen -t ed25519 -f "$PRIVATE_KEY" -N "" -C "dragon@dev-container"
    echo "New keys generated successfully."
fi