import unittest
import aim


class BackupCliParseTests(unittest.TestCase):
    def test_backup_parse(self):
        a = aim.build_parser().parse_args(["backup", "/tmp/bk", "--verify"])
        self.assertEqual(a.command, "backup")
        self.assertEqual(a.dest, "/tmp/bk")
        self.assertTrue(a.verify)

    def test_restore_parse(self):
        a = aim.build_parser().parse_args(["restore", "/tmp/bk", "--root", "ext", "--apply-env"])
        self.assertEqual(a.command, "restore")
        self.assertEqual(a.src, "/tmp/bk")
        self.assertEqual(a.root_id, "ext")
        self.assertTrue(a.apply_env)


if __name__ == "__main__":
    unittest.main()
