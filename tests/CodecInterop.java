import io.github.ivanbanny.protocol.VoxelTensor;
import io.github.ivanbanny.protocol.VoxelTensorCodec;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.List;
import net.minecraft.core.BlockPos;

public final class CodecInterop {
    public static void main(String[] args) throws Exception {
        byte[] encoded = Files.readAllBytes(Path.of(args[0]));
        byte[] wrongVersion = encoded.clone();
        wrongVersion[4] = 2;
        byte[] wrongPalette = encoded.clone();
        wrongPalette[wrongPalette.length - 4] = (byte) 0xff;
        wrongPalette[wrongPalette.length - 3] = (byte) 0xff;
        for (byte[] malformed : List.of(Arrays.copyOf(encoded, 31),
                Arrays.copyOf(encoded, encoded.length - 1), wrongVersion, wrongPalette)) {
            try {
                VoxelTensorCodec.decode(malformed);
                throw new AssertionError("Accepted malformed voxel tensor");
            } catch (IllegalArgumentException expected) {
            }
        }
        VoxelTensor tensor = VoxelTensorCodec.decode(encoded);
        Files.write(Path.of(args[1]), VoxelTensorCodec.encode(tensor));
        VoxelTensor.Sparse carve = new VoxelTensor.Sparse(
            new BlockPos(-1, 2, 3), 2, 1, 1,
            List.of("minecraft:air", "minecraft:stone"),
            new int[] {0, 1}, new int[] {1, 0});
        Files.write(Path.of(args[2]), VoxelTensorCodec.encode(carve));
    }
}
