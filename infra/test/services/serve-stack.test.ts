// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

import * as cdk from "aws-cdk-lib";
import * as dynamodb from "aws-cdk-lib/aws-dynamodb";
import * as ec2 from "aws-cdk-lib/aws-ec2";
import * as lambda from "aws-cdk-lib/aws-lambda";
import { Template, Match } from "aws-cdk-lib/assertions";
import { ServeStack } from "../../lib/stacks/services/serve-stack";
import { DEFAULT_RESOURCE_PREFIX, DEFAULT_ENV } from "../../lib/constants";

jest.mock("../../lib/utils/python-bundling", () => ({
  bundlePython: () =>
    lambda.Code.fromInline("def handler(event, context): pass"),
}));

const BASE_CONTEXT = {
  resource_prefix: DEFAULT_RESOURCE_PREFIX,
  env: DEFAULT_ENV,
  context_manager_image_uri:
    "123456789012.dkr.ecr.us-east-1.amazonaws.com/coa-dev:latest",
  "aws:cdk:bundling-stacks": [],
};

function createStack(contextOverrides: Record<string, string> = {}): Template {
  const app = new cdk.App({
    context: { ...BASE_CONTEXT, ...contextOverrides },
  });

  const depStack = new cdk.Stack(app, "DepStack", {
    env: { account: "123456789012", region: "us-east-1" },
  });
  const vpc = new ec2.Vpc(depStack, "Vpc", { maxAzs: 2 });
  const aossSg = new ec2.SecurityGroup(depStack, "AossSG", { vpc });
  const neptuneSg = new ec2.SecurityGroup(depStack, "NeptuneSG", { vpc });
  const lambdaSg = new ec2.SecurityGroup(depStack, "LambdaSG", { vpc });
  const mkTable = (id: string) =>
    new dynamodb.Table(depStack, id, {
      partitionKey: { name: "PK", type: dynamodb.AttributeType.STRING },
    });

  const stack = new ServeStack(app, "TestServe", {
    env: { account: "123456789012", region: "us-east-1" },
    vpc,
    aossSecurityGroup: aossSg,
    neptuneSecurityGroup: neptuneSg,
    lambdaSecurityGroup: lambdaSg,
    issuerUrl: "https://cognito-idp.us-east-1.amazonaws.com/us-east-1_test",
    clientId: "test-client-id",
    neptuneClusterArn:
      "arn:aws:neptune-db:us-east-1:123456789012:cluster:test-cluster/*",
    neptuneEndpoint: "test-cluster.cluster-abc.us-east-1.neptune.amazonaws.com",
    ontologyBucketArn: "arn:aws:s3:::coa-dev-ontology-artifacts",
    vkgEndpoint: "http://vkg.coa-dev-services.local:8080",
    rolesTable: mkTable("Roles"),
    resourceRoleMappingsTable: mkTable("RRM"),
  });

  return Template.fromStack(stack);
}

describe("ServeStack - Security Group egress", () => {
  describe("without JDBC peer CIDRs", () => {
    let template: Template;
    beforeAll(() => {
      template = createStack();
    });

    it("has local-VPC PostgreSQL egress on port 5432", () => {
      template.hasResourceProperties("AWS::EC2::SecurityGroup", {
        SecurityGroupEgress: Match.arrayWith([
          Match.objectLike({
            IpProtocol: "tcp",
            FromPort: 5432,
            ToPort: 5432,
            Description: Match.stringLikeRegexp("PostgreSQL.*within VPC"),
          }),
        ]),
      });
    });

    it("has local-VPC MySQL egress on port 3306", () => {
      template.hasResourceProperties("AWS::EC2::SecurityGroup", {
        SecurityGroupEgress: Match.arrayWith([
          Match.objectLike({
            IpProtocol: "tcp",
            FromPort: 3306,
            ToPort: 3306,
            Description: Match.stringLikeRegexp("MySQL.*within VPC"),
          }),
        ]),
      });
    });

    it("has local-VPC MSSQL egress on port 1433", () => {
      template.hasResourceProperties("AWS::EC2::SecurityGroup", {
        SecurityGroupEgress: Match.arrayWith([
          Match.objectLike({
            IpProtocol: "tcp",
            FromPort: 1433,
            ToPort: 1433,
            Description: Match.stringLikeRegexp("MSSQL.*within VPC"),
          }),
        ]),
      });
    });

    it("has local-VPC Redshift egress on port 5439", () => {
      template.hasResourceProperties("AWS::EC2::SecurityGroup", {
        SecurityGroupEgress: Match.arrayWith([
          Match.objectLike({
            IpProtocol: "tcp",
            FromPort: 5439,
            ToPort: 5439,
            Description: Match.stringLikeRegexp("Redshift.*within VPC"),
          }),
        ]),
      });
    });

    it("does NOT have peer-CIDR egress rules when no peering configured", () => {
      // No rule should reference a CIDR outside the VPC
      template.hasResourceProperties("AWS::EC2::SecurityGroup", {
        SecurityGroupEgress: Match.not(
          Match.arrayWith([
            Match.objectLike({
              Description: Match.stringLikeRegexp("peer network"),
            }),
          ]),
        ),
      });
    });
  });

  describe("with JDBC peer CIDRs configured", () => {
    let template: Template;
    beforeAll(() => {
      template = createStack({
        jdbc_peer_vpc_id: "vpc-peer123",
        jdbc_peer_cidrs: "10.20.0.0/16,10.30.0.0/16",
      });
    });

    it("adds PostgreSQL egress to each peer CIDR", () => {
      template.hasResourceProperties("AWS::EC2::SecurityGroup", {
        SecurityGroupEgress: Match.arrayWith([
          Match.objectLike({
            IpProtocol: "tcp",
            FromPort: 5432,
            ToPort: 5432,
            CidrIp: "10.20.0.0/16",
          }),
          Match.objectLike({
            IpProtocol: "tcp",
            FromPort: 5432,
            ToPort: 5432,
            CidrIp: "10.30.0.0/16",
          }),
        ]),
      });
    });

    it("adds MySQL egress to each peer CIDR", () => {
      template.hasResourceProperties("AWS::EC2::SecurityGroup", {
        SecurityGroupEgress: Match.arrayWith([
          Match.objectLike({
            IpProtocol: "tcp",
            FromPort: 3306,
            ToPort: 3306,
            CidrIp: "10.20.0.0/16",
          }),
          Match.objectLike({
            IpProtocol: "tcp",
            FromPort: 3306,
            ToPort: 3306,
            CidrIp: "10.30.0.0/16",
          }),
        ]),
      });
    });

    it("adds MSSQL egress to each peer CIDR", () => {
      template.hasResourceProperties("AWS::EC2::SecurityGroup", {
        SecurityGroupEgress: Match.arrayWith([
          Match.objectLike({
            IpProtocol: "tcp",
            FromPort: 1433,
            ToPort: 1433,
            CidrIp: "10.20.0.0/16",
          }),
          Match.objectLike({
            IpProtocol: "tcp",
            FromPort: 1433,
            ToPort: 1433,
            CidrIp: "10.30.0.0/16",
          }),
        ]),
      });
    });

    it("adds Redshift egress to each peer CIDR", () => {
      template.hasResourceProperties("AWS::EC2::SecurityGroup", {
        SecurityGroupEgress: Match.arrayWith([
          Match.objectLike({
            IpProtocol: "tcp",
            FromPort: 5439,
            ToPort: 5439,
            CidrIp: "10.20.0.0/16",
          }),
          Match.objectLike({
            IpProtocol: "tcp",
            FromPort: 5439,
            ToPort: 5439,
            CidrIp: "10.30.0.0/16",
          }),
        ]),
      });
    });
  });

  describe("with JDBC TGW CIDRs configured", () => {
    let template: Template;
    beforeAll(() => {
      template = createStack({
        jdbc_tgw_id: "tgw-abc123",
        jdbc_tgw_cidrs: "172.16.0.0/12",
      });
    });

    it("adds DB port egress to TGW CIDRs", () => {
      template.hasResourceProperties("AWS::EC2::SecurityGroup", {
        SecurityGroupEgress: Match.arrayWith([
          Match.objectLike({
            IpProtocol: "tcp",
            FromPort: 5432,
            ToPort: 5432,
            CidrIp: "172.16.0.0/12",
          }),
          Match.objectLike({
            IpProtocol: "tcp",
            FromPort: 3306,
            ToPort: 3306,
            CidrIp: "172.16.0.0/12",
          }),
        ]),
      });
    });
  });
});

describe("ServeStack - OE Monitoring", () => {
  let template: Template;
  beforeAll(() => {
    template = createStack();
  });

  it("emits a serve OE dashboard and Lambda alarms", () => {
    template.resourceCountIs("AWS::CloudWatch::Dashboard", 1);
    const alarms = template.findResources("AWS::CloudWatch::Alarm");
    expect(Object.keys(alarms).length).toBeGreaterThanOrEqual(1);
  });
});

describe("ServeStack - Athena S3 permissions", () => {
  let template: Template;
  beforeAll(() => {
    template = createStack();
  });

  /**
   * Collect every action string granted across all IAM policy statements whose
   * resource list mentions the given substring. Statements are keyed on the
   * bucket name rather than a policy logical id so the assertions survive
   * refactors that move statements between policies.
   */
  function actionsForResource(resourceSubstring: string): string[] {
    const policies = template.findResources("AWS::IAM::Policy");
    const actions: string[] = [];
    for (const policy of Object.values(policies)) {
      const statements = policy.Properties?.PolicyDocument?.Statement ?? [];
      for (const statement of statements) {
        const serialized = JSON.stringify(statement.Resource ?? "");
        if (!serialized.includes(resourceSubstring)) continue;
        const stmtActions = Array.isArray(statement.Action)
          ? statement.Action
          : [statement.Action];
        for (const action of stmtActions) {
          if (typeof action === "string") actions.push(action);
        }
      }
    }
    return actions;
  }

  // Athena writes result files with a multipart upload once they exceed the
  // single-PutObject threshold, and aborts the upload on failure. Small result
  // sets never exercise this, which is why the gap only surfaced against real
  // data — assert the actions explicitly so it cannot regress silently.
  it("grants multipart and delete actions on the Athena results bucket", () => {
    const actions = actionsForResource("athena-results");
    expect(actions).toContain("s3:AbortMultipartUpload");
    expect(actions).toContain("s3:ListMultipartUploadParts");
    expect(actions).toContain("s3:DeleteObject");
    expect(actions).toContain("s3:PutObject");
  });

  // The managed federated connector WRITES spill data here when results exceed
  // Lambda memory; a read-only policy fails only on large queries.
  it("grants write access on the Athena spill bucket", () => {
    const actions = actionsForResource("athena-spill");
    expect(actions).toContain("s3:PutObject");
    expect(actions).toContain("s3:GetObject");
    expect(actions).toContain("s3:AbortMultipartUpload");
    expect(actions).toContain("s3:ListMultipartUploadParts");
    expect(actions).toContain("s3:DeleteObject");
  });

  // Least privilege: the broadened S3 grants must stay scoped to the Athena
  // buckets and must not become a wildcard on all of S3.
  it("does not grant S3 wildcard actions", () => {
    const policies = template.findResources("AWS::IAM::Policy");
    for (const policy of Object.values(policies)) {
      const statements = policy.Properties?.PolicyDocument?.Statement ?? [];
      for (const statement of statements) {
        const stmtActions = Array.isArray(statement.Action)
          ? statement.Action
          : [statement.Action];
        expect(stmtActions).not.toContain("s3:*");
      }
    }
  });
});
