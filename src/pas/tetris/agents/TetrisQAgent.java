// java -cp ".;lib/*" edu.bu.pas.tetris.Main -p 1000 -t 20 -v 5 -g 0.99 -n 1e-3 --numUpdates 10 -b 128 -s | tee training.log

package src.pas.tetris.agents;


// SYSTEM IMPORTS
import java.util.Iterator;
import java.util.List;
import java.util.Random;
import java.util.Set;
import java.util.HashSet;
import java.util.ArrayList;


// JAVA PROJECT IMPORTS
import edu.bu.pas.tetris.agents.QAgent;
import edu.bu.pas.tetris.agents.TrainerAgent.GameCounter;
import edu.bu.pas.tetris.game.Board;
import edu.bu.pas.tetris.game.Game;
import edu.bu.pas.tetris.game.Game.GameView;
import edu.bu.pas.tetris.game.minos.Mino;
import edu.bu.pas.tetris.linalg.Matrix;
import edu.bu.pas.tetris.nn.Model;
import edu.bu.pas.tetris.nn.LossFunction;
import edu.bu.pas.tetris.nn.Optimizer;
import edu.bu.pas.tetris.nn.models.Sequential;
import edu.bu.pas.tetris.nn.layers.Dense; // fully connected layer
import edu.bu.pas.tetris.nn.layers.ReLU;  // some activations (below too)
import edu.bu.pas.tetris.nn.layers.Tanh;
import edu.bu.pas.tetris.nn.layers.Sigmoid;
import edu.bu.pas.tetris.training.data.Dataset;
import edu.bu.pas.tetris.utils.Pair;
import edu.bu.pas.tetris.game.Block;
import edu.bu.pas.tetris.utils.Coordinate;

public class TetrisQAgent extends QAgent {

    public static final double EXPLORATION_PROB = 0.05;

    private Random random;

    public TetrisQAgent(String name) {
        super(name);
        this.random = new Random(12345);
    }

    public Random getRandom() {
        return this.random;
    }

    @Override
    public Model initQFunction() {
        //final int numPixelsInImage = Board.NUM_ROWS * Board.NUM_COLS;
        //final int numFeatures = 5; // linesCleared, holes, bumpiness, maxHeight, supportRatio
        final int inputDim = 11;
        final int hiddenDim = 32;
        final int outDim = 1;

        Sequential qFunction = new Sequential();
        qFunction.add(new Dense(inputDim, hiddenDim));
        qFunction.add(new ReLU());
        qFunction.add(new Dense(hiddenDim, hiddenDim));
        qFunction.add(new ReLU());
        qFunction.add(new Dense(hiddenDim, outDim));
        return qFunction;
    }

    
    @Override
    public Matrix getQFunctionInput(final GameView game, final Mino move) {
        try {
            Board before = game.getBoard();
            Block[][] beforeState = before.getBoard();
            int[] originalHeights = new int[Board.NUM_COLS];
            int originalHoles = 0, originalBumpiness = 0, originalMaxHeight = 0;

            for (int x = 0; x < Board.NUM_COLS; x++) {
                boolean seen = false;
                for (int y = 0; y < Board.NUM_ROWS; y++) {
                    if (beforeState[y][x] != null) {
                        if (!seen) {
                            originalHeights[x] = Board.NUM_ROWS - y;
                            originalMaxHeight = Math.max(originalMaxHeight, originalHeights[x]);
                            seen = true;
                        }
                    } else if (seen) {
                        originalHoles++;
                    }
                }
                if (x > 0) originalBumpiness += Math.abs(originalHeights[x] - originalHeights[x - 1]);
            }

            Board after = new Board(before);
            after.addMino(move);
            List<Integer> clearedLines = after.clearFullLines();
            Block[][] afterState = after.getBoard();

            int linesCleared = clearedLines.size(), holes = 0, bumpiness = 0, maxHeight = 0;
            int supported = 0, total = 0, buried = 0;
            int[] heights = new int[Board.NUM_COLS];

            for (int x = 0; x < Board.NUM_COLS; x++) {
                boolean seen = false;
                for (int y = 0; y < Board.NUM_ROWS; y++) {
                    Block b = afterState[y][x];
                    if (b != null) {
                        if (!seen) {
                            heights[x] = Board.NUM_ROWS - y;
                            maxHeight = Math.max(maxHeight, heights[x]);
                            seen = true;
                        }
                        total++;
                        if (y == Board.NUM_ROWS - 1 || afterState[y + 1][x] != null) supported++;
                    } else if (seen) {
                        holes++;
                    }
                }
                if (x > 0) bumpiness += Math.abs(heights[x] - heights[x - 1]);
            }

            // Buried holes: empty cell with at least one block above it
            for (int x = 0; x < Board.NUM_COLS; x++) {
                boolean holeSeen = false;
                for (int y = Board.NUM_ROWS - 1; y >= 0; y--) {
                    if (afterState[y][x] == null) {
                        holeSeen = true;
                    } else if (holeSeen) {
                        buried++;
                    }
                }
            }

            int supportedPieceBlocks = 0;
            for (Block b : move.getBlocks()) {
                Coordinate c = b.getCoordinate();
                int x = c.getXCoordinate(), y = c.getYCoordinate();
                if (y == Board.NUM_ROWS - 1 || afterState[y + 1][x] != null) supportedPieceBlocks++;
            }

            // Final features
            double deltaHoles = holes - originalHoles;
            double deltaBumpiness = bumpiness - originalBumpiness;
            double deltaMaxHeight = maxHeight - originalMaxHeight;
            double supportRatio = total > 0 ? (double) supported / total : 0.0;
            double pieceSupportRatio = (double) supportedPieceBlocks / move.getBlocks().length;
            double heightRatio = (double) maxHeight / Board.NUM_ROWS;

            // Vector: lines, holes, buried, bumpiness, maxH, support, pieceSupport, dHoles, dBumpiness, dMaxH, heightRatio
            Matrix input = Matrix.zeros(1, 11);
            input.set(0, 0, linesCleared);
            input.set(0, 1, holes);
            input.set(0, 2, buried);
            input.set(0, 3, bumpiness);
            input.set(0, 4, maxHeight);
            input.set(0, 5, supportRatio);
            input.set(0, 6, pieceSupportRatio);
            input.set(0, 7, deltaHoles);
            input.set(0, 8, deltaBumpiness);
            input.set(0, 9, deltaMaxHeight);
            input.set(0, 10, heightRatio);

            return input;
        } catch (Exception e) {
            e.printStackTrace();
            System.exit(-1);
            return null;
        }
    }


    @Override
    public boolean shouldExplore(final GameView game, final GameCounter gameCounter) {
        final double MAX_PROB = 0.8;
        final double MIN_PROB = 0.05;
        final double DECAY_RATE = 0.0005;

        double gameIdx = gameCounter.getCurrentGameIdx();
        double explorationProb = MIN_PROB + (MAX_PROB - MIN_PROB) * Math.exp(-DECAY_RATE * gameIdx);

        boolean explore = this.getRandom().nextDouble() < explorationProb;

        if (!explore) {
            //System.out.printf("[Policy] NOT exploring at game=%d | prob=%.3f%n", (int) gameIdx, explorationProb);
        }

        return explore;
    }


    @Override
    public Mino getExplorationMove(final GameView gameView) {
        List<Mino> possibleMoves = gameView.getFinalMinoPositions();
        Board originalBoard = gameView.getBoard();
        Mino bestMove = null;
        double bestScore = Double.NEGATIVE_INFINITY;
        String bestContributor = "";

        // Analyze original board
        int[] originalHeights = new int[Board.NUM_COLS];
        int originalHoles = 0, originalBumpiness = 0, originalMaxHeight = 0;
        Block[][] originalState = originalBoard.getBoard();
        for (int x = 0; x < Board.NUM_COLS; x++) {
            boolean seenBlock = false;
            for (int y = 0; y < Board.NUM_ROWS; y++) {
                if (originalState[y][x] != null) {
                    if (!seenBlock) {
                        originalHeights[x] = Board.NUM_ROWS - y;
                        originalMaxHeight = Math.max(originalMaxHeight, originalHeights[x]);
                        seenBlock = true;
                    }
                } else if (seenBlock) {
                    originalHoles++;
                }
            }
            if (x > 0) originalBumpiness += Math.abs(originalHeights[x] - originalHeights[x - 1]);
        }

        // === Target Well Detection (only if piece is an I-block) ===
        int targetWellColumn = -1;
        if (!possibleMoves.isEmpty() && possibleMoves.get(0).getType() == Mino.MinoType.I) {
            int[] wellHeights = new int[Board.NUM_COLS];
            for (int x = 0; x < Board.NUM_COLS; x++) {
                for (int y = 0; y < Board.NUM_ROWS; y++) {
                    if (originalState[y][x] != null) {
                        wellHeights[x] = Board.NUM_ROWS - y;
                        break;
                    }
                }
            }

            for (int x = 0; x < Board.NUM_COLS; x++) {
                int left = (x > 0) ? wellHeights[x - 1] : -1;
                int right = (x < Board.NUM_COLS - 1) ? wellHeights[x + 1] : -1;

                boolean validWell =
                    (x > 0 && x < Board.NUM_COLS - 1 && wellHeights[x] < left && wellHeights[x] < right) || // inner well
                    (x == 0 && wellHeights[x] < right) || // left edge well
                    (x == Board.NUM_COLS - 1 && wellHeights[x] < left); // right edge well

                int depth = Math.min(
                    x == 0 ? right : (x == Board.NUM_COLS - 1 ? left : Math.min(left, right)),
                    Board.NUM_ROWS
                ) - wellHeights[x];

                if (validWell && depth >= 3) {
                    targetWellColumn = x;
                    break;
                }
            }

        }

        // === Move Evaluation ===
        for (Mino move : possibleMoves) {
            Board simulatedBoard = new Board(originalBoard);
            simulatedBoard.addMino(move);
            List<Integer> clearedLines = simulatedBoard.clearFullLines();
            if (clearedLines.size() == 4) {
                //System.out.println("[Exploration] Tetris detected! Returning move immediately.");
                return move;
            }


            int linesCleared = clearedLines.size(), holes = 0, bumpiness = 0, maxHeight = 0;
            int supportedBlocks = 0, totalBlocks = 0;
            int[] heights = new int[Board.NUM_COLS];
            Block[][] newState = simulatedBoard.getBoard();

            for (int x = 0; x < Board.NUM_COLS; x++) {
                boolean seenBlock = false;
                for (int y = 0; y < Board.NUM_ROWS; y++) {
                    Block cell = newState[y][x];
                    if (cell != null) {
                        if (!seenBlock) {
                            heights[x] = Board.NUM_ROWS - y;
                            maxHeight = Math.max(maxHeight, heights[x]);
                            seenBlock = true;
                        }
                        totalBlocks++;
                        if (y == Board.NUM_ROWS - 1 || newState[y + 1][x] != null) supportedBlocks++;
                    } else if (seenBlock) {
                        holes++;
                    }
                }
                if (x > 0) bumpiness += Math.abs(heights[x] - heights[x - 1]);
            }

            int buriedHoles = 0;
            for (int x = 0; x < Board.NUM_COLS; x++) {
                boolean holeSeen = false;
                for (int y = Board.NUM_ROWS - 1; y >= 0; y--) {
                    if (newState[y][x] == null) {
                        holeSeen = true;
                    } else if (holeSeen) {
                        buriedHoles++;
                    }
                }
            }

            double supportRatio = (totalBlocks > 0) ? (double) supportedBlocks / totalBlocks : 0.0;

            Block[] placedBlocks = move.getBlocks();
            int supportedPieceBlocks = 0;
            for (Block b : placedBlocks) {
                Coordinate coord = b.getCoordinate();
                int x = coord.getXCoordinate(), y = coord.getYCoordinate();
                if (y == Board.NUM_ROWS - 1 || newState[y + 1][x] != null) supportedPieceBlocks++;
            }

            double pieceSupportRatio = (double) supportedPieceBlocks / placedBlocks.length;

            double heightRatio = (double) maxHeight / Board.NUM_ROWS;
            double lineClearBonus = (heightRatio > 0.6)
                ? (linesCleared == 1 ? 2.0 : linesCleared == 2 ? 4.0 : linesCleared == 3 ? 10.0 : 0.0)
                : (linesCleared == 1 ? 0.5 : linesCleared == 2 ? 1.0 : linesCleared == 3 ? 5.0 : 0.0);

            double deltaHoles = holes - originalHoles;
            double deltaBumpiness = bumpiness - originalBumpiness;
            double deltaMaxHeight = maxHeight - originalMaxHeight;

            if (targetWellColumn != -1 && move.getType() == Mino.MinoType.I) {
                Mino rotatedMino = move.rotateTo(Mino.Orientation.B);
                int currentX = rotatedMino.getPivotBlockCoordinate().getXCoordinate();
                int shiftAmount = targetWellColumn - currentX;

                for (int i = 0; i < Math.abs(shiftAmount); i++) {
                    rotatedMino = shiftAmount > 0 ? rotatedMino.moveRight() : rotatedMino.moveLeft();
                }

                Mino tempMino = rotatedMino.moveDown();
                while (gameView.getBoard().isLegalPosition(tempMino.getBlocks()) &&
                    !blocksOverlap(gameView.getBoard(), tempMino)) {
                    rotatedMino = tempMino;
                    tempMino = rotatedMino.moveDown();
                }

                if (!gameView.getBoard().isLegalPosition(rotatedMino.getBlocks()) ||
                    blocksOverlap(gameView.getBoard(), rotatedMino)) {
                    rotatedMino = shiftMinoUp(rotatedMino);
                }

                if (gameView.getBoard().isLegalPosition(rotatedMino.getBlocks()) &&
                    !blocksOverlap(gameView.getBoard(), rotatedMino)) {
                    return rotatedMino;
                } else {
                    
                    //System.out.println("[Well Fill] Skipping bonus due to illegal rotated mino.");
                    }

            }

            double deltaBumpinessPenalty = deltaBumpiness < 0 ? 0.3 * -deltaBumpiness : 0.2 * -deltaBumpiness;

            double[] weights = {
                1.0 * lineClearBonus,
                0.6 * pieceSupportRatio,
                0.5 * supportRatio,
                -2.5 * deltaHoles,
                deltaBumpinessPenalty,
                -0.3 * deltaMaxHeight,
                -0.1 * holes,
                -0.07 * bumpiness,
                -0.1 * maxHeight,
                Math.max(-0.8, -0.4 * buriedHoles)
            };

            String[] labels = {
                "lineClearBonus", "pieceSupportRatio", "supportRatio",
                "deltaHoles", "deltaBumpiness", "deltaMaxHeight", "holes",
                "bumpiness", "maxHeight", "buriedHoles"
            };

            double score = 0.0;
            for (double w : weights) score += w;

            if (score > bestScore) {
                bestScore = score;
                bestMove = move;
                double maxVal = Math.abs(weights[0]);
                int maxIdx = 0;
                for (int i = 1; i < weights.length; i++) {
                    if (Math.abs(weights[i]) > maxVal) {
                        maxVal = Math.abs(weights[i]);
                        maxIdx = i;
                    }
                }
                bestContributor = String.format("%s = %.3f", labels[maxIdx], weights[maxIdx]);
            }
        }

        //System.out.printf("[Exploration] score = %.2f\n", bestScore);
        return bestMove;
    }


    private Mino shiftMinoUp(Mino mino) {
        Coordinate pivot = mino.getPivotBlockCoordinate();
        Coordinate shiftedPivot = new Coordinate(pivot.getXCoordinate(), pivot.getYCoordinate() - 1);
        return Mino.create(mino.getType(), shiftedPivot, mino.getOrientation());
    }

    private boolean blocksOverlap(Board board, Mino mino) {
        for (Block block : mino.getBlocks()) {
            Coordinate coord = block.getCoordinate();
            int x = coord.getXCoordinate();
            int y = coord.getYCoordinate();
            if (board.getBoard()[y][x] != null) {
                return true;
            }
        }
        return false;
    }
    
    private void printBoardState(Board board) {
        Block[][] grid = board.getBoard();
        System.out.println("[Board State] (Top to Bottom)");
        for (int y = 0; y < Board.NUM_ROWS; y++) {
            System.out.printf("%2d | ", y);
            for (int x = 0; x < Board.NUM_COLS; x++) {
                System.out.print(grid[y][x] != null ? "X " : ". ");
            }
            System.out.println();
        }
        System.out.print("    ");
        for (int x = 0; x < Board.NUM_COLS; x++) {
            System.out.printf("%d ", x);
        }
        System.out.println("\n");
    }

    @Override
    public void trainQFunction(Dataset dataset,
                               LossFunction lossFunction,
                               Optimizer optimizer,
                               long numUpdates) {
        for (int epochIdx = 0; epochIdx < numUpdates; ++epochIdx) {
            dataset.shuffle();
            Iterator<Pair<Matrix, Matrix>> batchIterator = dataset.iterator();

            while (batchIterator.hasNext()) {
                Pair<Matrix, Matrix> batch = batchIterator.next();
                try {
                    Matrix YHat = this.getQFunction().forward(batch.getFirst());
                    optimizer.reset();
                    this.getQFunction().backwards(batch.getFirst(),
                                                  lossFunction.backwards(YHat, batch.getSecond()));
                    optimizer.step();
                } catch (Exception e) {
                    e.printStackTrace();
                    System.exit(-1);
                }
            }
        }
    }


    @Override
    public double getReward(final GameView game) {
        Board originalBoard = game.getBoard();
        Block[][] originalState = originalBoard.getBoard();

        // Compute original board features
        int[] originalHeights = new int[Board.NUM_COLS];
        int originalHoles = 0, originalBumpiness = 0, originalMaxHeight = 0;

        for (int x = 0; x < Board.NUM_COLS; x++) {
            boolean seenBlock = false;
            for (int y = 0; y < Board.NUM_ROWS; y++) {
                if (originalState[y][x] != null) {
                    if (!seenBlock) {
                        originalHeights[x] = Board.NUM_ROWS - y;
                        originalMaxHeight = Math.max(originalMaxHeight, originalHeights[x]);
                        seenBlock = true;
                    }
                } else if (seenBlock) {
                    originalHoles++;
                }
            }
            if (x > 0) {
                originalBumpiness += Math.abs(originalHeights[x] - originalHeights[x - 1]);
            }
        }

        // Simulate board after turn
        Board simulatedBoard = new Board(originalBoard);
        simulatedBoard.clearFullLines();
        Block[][] newState = simulatedBoard.getBoard();

        int linesCleared = 0, holes = 0, bumpiness = 0, maxHeight = 0;
        int supportedBlocks = 0, totalBlocks = 0;
        int buriedHoles = 0;
        int[] heights = new int[Board.NUM_COLS];

        for (int x = 0; x < Board.NUM_COLS; x++) {
            boolean seenBlock = false;
            for (int y = 0; y < Board.NUM_ROWS; y++) {
                Block cell = newState[y][x];
                if (cell != null) {
                    if (!seenBlock) {
                        heights[x] = Board.NUM_ROWS - y;
                        maxHeight = Math.max(maxHeight, heights[x]);
                        seenBlock = true;
                    }
                    totalBlocks++;
                    if (y == Board.NUM_ROWS - 1 || newState[y + 1][x] != null) supportedBlocks++;
                } else if (seenBlock) {
                    holes++;
                }
            }
            if (x > 0) bumpiness += Math.abs(heights[x] - heights[x - 1]);
        }

        for (int x = 0; x < Board.NUM_COLS; x++) {
            boolean holeSeen = false;
            for (int y = Board.NUM_ROWS - 1; y >= 0; y--) {
                if (newState[y][x] == null) {
                    holeSeen = true;
                } else if (holeSeen) {
                    buriedHoles++;
                }
            }
        }

        double supportRatio = (totalBlocks > 0) ? (double) supportedBlocks / totalBlocks : 0.0;

        // Estimate piece support ratio (for this turn)
        double pieceSupportRatio = supportRatio; // Approximate since we can't track piece only

        double heightRatio = (double) maxHeight / Board.NUM_ROWS;

        double lineClearBonus = (heightRatio > 0.6)
            ? (linesCleared == 1 ? 2.0 : linesCleared == 2 ? 4.0 : linesCleared == 3 ? 10.0 : 0.0)
            : (linesCleared == 1 ? 0.5 : linesCleared == 2 ? 1.0 : linesCleared == 3 ? 5.0 : 0.0);

        // Deltas
        double deltaHoles = holes - originalHoles;
        double deltaBumpiness = bumpiness - originalBumpiness;
        double deltaMaxHeight = maxHeight - originalMaxHeight;

        // Final reward
        double reward =
            //+ 1.0 * game.getTotalScore()
            + 5.0 * lineClearBonus
            + 1.0 * pieceSupportRatio
            + 0.5 * supportRatio
            - 2.0 * deltaHoles
            + 0.3 * (-deltaBumpiness)
            + 1.0 * (-deltaMaxHeight)
            - 0.02 * holes
            - 0.02 * bumpiness
            - 0.05 * maxHeight
            - Math.min(0.7, 0.4 * buriedHoles);


        //System.out.printf("[Reward Debug] reward = %.3f\n", reward, game.getScoreThisTurn(), game.getTotalScore());

        return reward;
    }



    /**
     * Override updateQFunction if you want to do real-time TD(0) learning here.
     * Otherwise training happens in batch above.
     */
}